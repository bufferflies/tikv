// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.
//
// Keyspace mapping management for id <-> name
// Responsible for fetching from PD, caching locally, and providing lookup APIs
// Usage:
// 1. Initialize the keyspace manager with `init_keyspace_manager`.
// 2. Use `to_keyspace_name` to convert keyspace id to keyspace name.

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
};

use lazy_static::lazy_static;
use tikv_util::warn;

use crate::PdClient;

thread_local! {
    // Thread local cache for all keyspaces
    static LOCAL_KEYSPACE_CACHE: RefCell<HashMap<u32, Arc<String>>> = RefCell::new(HashMap::new());
}

/// Manages keyspace ID to name mapping with multi-level caching strategy.
///
/// This struct provides efficient keyspace name lookups by maintaining:
/// 1. A global cache shared across all threads
/// 2. Thread-local caches for faster access
/// 3. Background fetching from PD for cache misses
///
/// The manager uses a two-level caching approach:
/// - Thread-local cache: Provides fastest access for frequently accessed
///   keyspaces
/// - Global cache: Shared across all threads, preloaded during initialization
///
/// When a keyspace is not found in either cache, a background task is spawned
/// to fetch it from PD and update the global cache for future requests.
struct KeyspaceManager {
    global_cache: RwLock<HashMap<u32, Arc<String>>>,
    // This flag prevents duplicate PD requests for the same keyspace ID in the same thread or
    // different threads. It's overkill that this also prevents requests for different keyspace
    // IDs, but we accept this because:
    // 1. Cache misses are infrequent since existing keyspaces are preloaded and new keyspace
    //    creation is infrequent.
    // 2. The implementation simplicity justifies the trade-off
    pd_fetch_in_progress: AtomicBool,
    pd_client: Arc<dyn PdClient>,
}

lazy_static! {
    static ref KEYSPACE_MANAGER: RwLock<Option<Arc<KeyspaceManager>>> = RwLock::new(None);
}

impl KeyspaceManager {
    fn new(pd_client: Arc<dyn PdClient>) -> Self {
        Self {
            global_cache: RwLock::new(HashMap::new()),
            pd_fetch_in_progress: AtomicBool::new(false),
            pd_client,
        }
    }

    fn get_keyspace_name(self: &Arc<Self>, keyspace_id: u32) -> Option<Arc<String>> {
        // First try thread local cache
        if let Some(name) =
            LOCAL_KEYSPACE_CACHE.with(|cache| cache.borrow().get(&keyspace_id).cloned())
        {
            return Some(name);
        }

        // Check global cache
        if let Some(name) = self.global_cache.read().unwrap().get(&keyspace_id).cloned() {
            // Update thread local cache
            LOCAL_KEYSPACE_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                cache.insert(keyspace_id, name.clone());
            });
            return Some(name);
        }

        // If not in any cache and no other thread is fetching, spawn a background task
        // to load from PD
        if !self.pd_fetch_in_progress.load(Ordering::Acquire)
            && self
                .pd_fetch_in_progress
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            let pd_client = self.pd_client.clone();
            let manager = self.clone();
            std::thread::spawn(move || {
                // Use get_all_keyspaces with start_id and limit=1 to fetch single keyspace
                if let Ok(keyspaces) = pd_client.get_all_keyspaces(Some(keyspace_id), Some(1)) {
                    // Only update cache if we got exactly one keyspace and its ID matches
                    if let Some(meta) = keyspaces.first() {
                        if meta.get_id() == keyspace_id {
                            let name = meta.get_name().to_string();
                            // Update global cache
                            manager
                                .global_cache
                                .write()
                                .unwrap()
                                .insert(keyspace_id, Arc::new(name));
                        } else {
                            warn!(
                                "keyspace-id not found in PD: requested {}, got {}",
                                keyspace_id,
                                meta.get_id()
                            );
                        }
                    }
                }
                manager.pd_fetch_in_progress.store(false, Ordering::Release);
            });
        }

        // Return none for this request
        None
    }
}

/// Initialize the global keyspace manager.
/// This should be called once at startup.
pub fn init_keyspace_manager(pd_client: Arc<dyn PdClient>) {
    // Load all keyspaces from PD in one RPC call to amortize the RPC overhead.
    let mut map = HashMap::new();
    let all_keyspaces = pd_client.get_all_keyspaces(None, None).unwrap_or_default();
    for keyspace in all_keyspaces {
        map.insert(keyspace.get_id(), Arc::new(keyspace.get_name().to_string()));
    }

    let keyspace_manager = Arc::new(KeyspaceManager::new(pd_client.clone()));
    {
        let mut global_cache = keyspace_manager.global_cache.write().unwrap();
        global_cache.extend(map);
    }

    let mut manager = KEYSPACE_MANAGER.write().unwrap();
    *manager = Some(keyspace_manager);
}

/// Convert keyspace id to keyspace name.
/// This is the main public interface for looking up keyspace names.
///
/// - Returns `Some(Arc<String>)` if the keyspace name is in the cache.
/// - Returns `None` if the keyspace name is not in the cache. In this case, the
///   keyspace name will be fetched from PD asynchronously in the background
///   thread and then added to the cache.
pub fn to_keyspace_name(keyspace_id: u32) -> Option<Arc<String>> {
    KEYSPACE_MANAGER
        .read()
        .ok()?
        .as_ref()?
        .get_keyspace_name(keyspace_id)
}

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicU32, Mutex};

    use kvproto::keyspacepb::KeyspaceMeta;
    use security::{GetSecurityManager, SecurityManager};

    use super::*;
    use crate::Result;

    // Used to ensure test isolation.
    lazy_static! {
        static ref TEST_LOCK: Mutex<()> = Mutex::new(());
    }

    /// Reset all global state to ensure test isolation. This is necessary
    /// because:
    /// 1. The module uses global variables (KEYSPACE_MANAGER, etc.)
    /// 2. Tests run in parallel by default, which can cause state interference
    /// 3. Each test needs a clean state to avoid side effects from other tests
    fn reset_global_state_for_test_isolation() {
        // Clear manager
        let mut manager = KEYSPACE_MANAGER.write().unwrap();
        if let Some(m) = &*manager {
            m.global_cache.write().unwrap().clear();
            m.pd_fetch_in_progress.store(false, Ordering::SeqCst);
        }
        *manager = None;
        LOCAL_KEYSPACE_CACHE.with(|cache| {
            cache.borrow_mut().clear();
        });
    }

    struct MockPdClient {
        keyspaces: Vec<KeyspaceMeta>,
        call_count: AtomicU32,
        first_call_empty: bool,
        sleep_on_call: bool,
    }

    impl MockPdClient {
        fn new(keyspaces: Vec<KeyspaceMeta>, first_call_empty: bool, sleep_on_call: bool) -> Self {
            let mut sorted_keyspaces = keyspaces;
            sorted_keyspaces.sort_by_key(|k| k.get_id());
            Self {
                keyspaces: sorted_keyspaces,
                call_count: AtomicU32::new(0),
                first_call_empty,
                sleep_on_call,
            }
        }

        fn get_call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl GetSecurityManager for MockPdClient {
        fn get_security_mgr(&self) -> Arc<SecurityManager> {
            Arc::new(SecurityManager::default())
        }
    }

    impl PdClient for MockPdClient {
        fn get_all_keyspaces(
            &self,
            start_id: Option<u32>,
            limit: Option<u32>,
        ) -> Result<Vec<KeyspaceMeta>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            // First call returns empty result
            if self.first_call_empty && self.get_call_count() == 1 {
                return Ok(vec![]);
            }

            if self.sleep_on_call {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }

            let start = start_id.unwrap_or(0);
            let limit = limit.unwrap_or(u32::MAX) as usize;

            // Find the first keyspace with ID >= start_id
            let result: Vec<_> = self
                .keyspaces
                .iter()
                .filter(|k| k.get_id() >= start)
                .take(limit)
                .cloned()
                .collect();

            Ok(result)
        }
    }

    fn create_test_keyspace(id: u32, name: &str) -> KeyspaceMeta {
        let mut meta = KeyspaceMeta::default();
        meta.set_id(id);
        meta.set_name(name.to_string());
        meta
    }

    #[test]
    fn test_keyspace_manager_initialization() {
        let _lock = TEST_LOCK.lock().unwrap();
        reset_global_state_for_test_isolation();

        let keyspaces = vec![
            create_test_keyspace(1, "test1"),
            create_test_keyspace(2, "test2"),
        ];
        let mock_client = Arc::new(MockPdClient::new(keyspaces, false, false));

        init_keyspace_manager(mock_client.clone());

        // Test to_keyspace_name
        assert_eq!(*to_keyspace_name(1).unwrap(), "test1");
        assert_eq!(*to_keyspace_name(2).unwrap(), "test2");

        // Verify only one call to get_all_keyspaces was made
        assert_eq!(mock_client.get_call_count(), 1);
    }

    #[test]
    fn test_keyspace_cache() {
        let _lock = TEST_LOCK.lock().unwrap();
        reset_global_state_for_test_isolation();

        let keyspaces = vec![
            create_test_keyspace(1, "test1"),
            create_test_keyspace(2, "test2"),
        ];
        let mock_client = Arc::new(MockPdClient::new(keyspaces, false, false));

        init_keyspace_manager(mock_client.clone());

        // First call should hit cache
        assert_eq!(*to_keyspace_name(1).unwrap(), "test1");
        assert_eq!(*to_keyspace_name(1).unwrap(), "test1");
    }

    #[test]
    fn test_unknown_keyspace() {
        let _lock = TEST_LOCK.lock().unwrap();
        reset_global_state_for_test_isolation();

        let keyspaces = vec![
            create_test_keyspace(1, "test1"),
            create_test_keyspace(3, "test3"),
        ];
        let mock_client = Arc::new(MockPdClient::new(keyspaces, false, false));

        init_keyspace_manager(mock_client.clone());

        // Test unknown keyspace
        assert_eq!(to_keyspace_name(2), None);
        // Wait for background fetch task to complete
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(to_keyspace_name(2), None);
        assert_eq!(to_keyspace_name(4), None);
        // Wait for background fetch task to complete
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(to_keyspace_name(4), None);
    }

    #[test]
    fn test_keyspace_lookup_with_delay() {
        let _lock = TEST_LOCK.lock().unwrap();
        reset_global_state_for_test_isolation();

        let keyspaces = vec![
            create_test_keyspace(1, "test1"),
            create_test_keyspace(2, "test2"),
        ];
        let mock_client = Arc::new(MockPdClient::new(keyspaces, true, false));

        init_keyspace_manager(mock_client.clone());

        // First lookup should return None
        assert_eq!(to_keyspace_name(1), None);

        // Wait a bit for background task to complete
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Second lookup should return the actual name
        assert_eq!(*to_keyspace_name(1).unwrap(), "test1");

        // Verify that get_all_keyspaces was called twice: first for preload, then for
        // fetch specific keyspace
        assert_eq!(mock_client.get_call_count(), 2);
    }

    #[test]
    fn test_pd_fetch_in_progress() {
        let _lock = TEST_LOCK.lock().unwrap();
        reset_global_state_for_test_isolation();

        let keyspaces = vec![
            create_test_keyspace(1, "test1"),
            create_test_keyspace(2, "test2"),
        ];
        let mock_client = Arc::new(MockPdClient::new(keyspaces, false, true));

        init_keyspace_manager(mock_client.clone());

        // Spawn multiple threads to request the same unknown keyspace
        let handles: Vec<_> = (0..10)
            .map(|_| std::thread::spawn(move || to_keyspace_name(999)))
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            assert_eq!(handle.join().unwrap(), None);
        }

        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify that get_all_keyspaces was called only three times: first for preload,
        // then for fetch specific keyspace
        assert_eq!(mock_client.get_call_count(), 2);
    }
}
