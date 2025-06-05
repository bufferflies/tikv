// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    ops::Deref,
    sync::atomic::{AtomicUsize, Ordering},
};

use parking_lot::{Mutex, RwLock};

/// An alternative util to simple RwLock. It allows writing not blocking
/// reading, at the expense of linearizability between reads and writes.
///
/// This is suitable for use cases where atomic storing and loading is expected,
/// but atomic variables is not applicable due to the inner type larger than 8
/// bytes. When writing is in progress, readings will get the previous value.
/// Writes will block each other, and fast and frequent writes may also block or
/// be blocked by slow reads.
///
/// This is ported from the HealthController module from the opensource
/// repository: https://github.com/tikv/tikv/blob/43e63b5614c96119e4126d8c2a29342e47b95d3d/components/health_controller/src/lib.rs#L275
pub struct RollingRetriever<T> {
    content: [RwLock<T>; 2],
    current_index: AtomicUsize,
    write_mutex: Mutex<()>,
}

impl<T: Default> RollingRetriever<T> {
    pub fn new() -> Self {
        Self::new_with(T::default)
    }
}

impl<T: Default> Default for RollingRetriever<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RollingRetriever<T> {
    pub fn new_with(initializer: impl Fn() -> T) -> Self {
        Self {
            content: [RwLock::new(initializer()), RwLock::new(initializer())],
            current_index: AtomicUsize::new(0),
            write_mutex: Mutex::new(()),
        }
    }

    #[inline]
    pub fn put(&self, new_value: T) {
        self.put_with(|| new_value)
    }

    fn put_with(&self, f: impl FnOnce() -> T) {
        let _write_guard = self.write_mutex.lock();
        // Update the item that is not the currently active one
        let index = self.current_index.load(Ordering::Acquire) ^ 1;

        let mut data_guard = self.content[index].write();
        *data_guard = f();

        drop(data_guard);
        self.current_index.store(index, Ordering::Release);
    }

    pub fn read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let index = self.current_index.load(Ordering::Acquire);
        let guard = self.content[index].read();
        f(guard.deref())
    }
}

impl<T: Clone> RollingRetriever<T> {
    pub fn get_cloned(&self) -> T {
        self.read(|r| r.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            mpsc::{sync_channel, RecvTimeoutError},
            Arc,
        },
        time::Duration,
    };

    use super::*;
    #[test]
    fn test_rolling_retriever() {
        let r = Arc::new(RollingRetriever::<u64>::new());
        assert_eq!(r.get_cloned(), 0);

        for i in 1..=10 {
            r.put(i);
            assert_eq!(r.get_cloned(), i);
        }

        // Writing doesn't block reading.
        let r1 = r.clone();
        let (write_continue_tx, rx) = sync_channel(0);
        let write_handle = std::thread::spawn(move || {
            r1.put_with(move || {
                rx.recv().unwrap();
                11
            })
        });
        for _ in 1..10 {
            std::thread::sleep(Duration::from_millis(5));
            assert_eq!(r.get_cloned(), 10)
        }
        write_continue_tx.send(()).unwrap();
        write_handle.join().unwrap();
        assert_eq!(r.get_cloned(), 11);

        // Writing block each other.
        let r1 = r.clone();
        let (write1_tx, rx1) = sync_channel(0);
        let write1_handle = std::thread::spawn(move || {
            r1.put_with(move || {
                // Receive once for notifying lock acquired.
                rx1.recv().unwrap();
                // Receive again to be notified ready to continue.
                rx1.recv().unwrap();
                12
            })
        });
        write1_tx.send(()).unwrap();
        let r1 = r.clone();
        let (write2_tx, rx2) = sync_channel(0);
        let write2_handle = std::thread::spawn(move || {
            r1.put_with(move || {
                write2_tx.send(()).unwrap();
                13
            })
        });
        // Write 2 cannot continue as blocked by write 1.
        assert_eq!(
            rx2.recv_timeout(Duration::from_millis(50)).unwrap_err(),
            RecvTimeoutError::Timeout
        );
        // Continue write1
        write1_tx.send(()).unwrap();
        write1_handle.join().unwrap();
        assert_eq!(r.get_cloned(), 12);
        // Continue write2
        rx2.recv().unwrap();
        write2_handle.join().unwrap();
        assert_eq!(r.get_cloned(), 13);
    }
}
