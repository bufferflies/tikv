// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

//! This module startups all the components of a TiKV server.
//!
//! It is responsible for reading from configs, starting up the various server
//! components, and handling errors (mostly by aborting and reporting to the
//! user).
//!
//! The entry point is `run_tikv`.
//!
//! Components are often used to initialize other components, and/or must be
//! explicitly stopped. We keep these components in the `TiKVServer` struct.

use std::{
    env, fmt,
    fs::{self, File},
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{atomic::AtomicU64, Arc, Once},
    u64,
};

use api_version::{dispatch_api_version, KvFormat};
use concurrency_manager::ConcurrencyManager;
use engine_rocks::from_rocks_compression_type;
use engine_traits::{KvEngine, RaftEngine, CF_DEFAULT, CF_WRITE};
use file_system::{
    BytesFetcher, IoRateLimitMode, IoRateLimiter, MetricsManager as IoMetricsManager,
};
use fs2::FileExt;
use futures::executor::block_on;
use grpcio::{EnvBuilder, Environment};
use kvengine::dfs::Dfs;
use kvproto::{
    brpb::create_backup, deadlock::create_deadlock, import_sstpb_grpc::create_import_sst,
    raft_serverpb::StoreIdent,
};
use overload_protector::{OverloadProtector, OverloadProtectorWorker};
use pd_client::{pd_control::PdControl, PdClient, RpcClient, INVALID_ID};
use protobuf::Message;
use raftstore::{
    coprocessor::{
        BoxConsistencyCheckObserver, ConsistencyCheckMethod, CoprocessorHost,
        RawConsistencyCheckObserver,
    },
    RegionInfoAccessor,
};
use rfengine::{RfEngine, STORE_IDENT_KEY};
use rfstore::{
    store::{
        BlackList, Engines, LocalReader, MetaChangeListener, PdIdAllocator, RaftBatchSystem,
        StoreMeta, StoreMsg, PENDING_MSG_CAP,
    },
    RaftRouter, ServerRaftStoreRouter,
};
use security::SecurityManager;
use sst_importer::SstImporter;
use tikv::{
    config::{ConfigController, TikvConfig},
    coprocessor, coprocessor_v2,
    read_pool::{build_yatp_read_pool, ReadPool},
    server::{
        config::Config as ServerConfig, lock_manager::LockManager, raftkv::ReplicaReadLockChecker,
        CPU_CORES_QUOTA_GAUGE, DEFAULT_CLUSTER_ID, GRPC_THREAD_PREFIX,
    },
    storage::{
        self,
        mvcc::MvccConsistencyCheckObserver,
        txn::flow_controller::{EngineFlowController, FlowController},
    },
};
use tikv_kv::Engine;
use tikv_util::{
    check_environment_variables,
    config::{ensure_dir_exist, ReadableDuration, VersionTrack},
    get_panic_region_count, mpsc,
    quota_limiter::{QuotaLimitConfigManager, QuotaLimiter},
    sys::{register_memory_usage_high_water, thread::ThreadBuildWrapper, SysQuota},
    thread_group::GroupProperties,
    time::{Duration, Instant, Monitor},
    worker::{Builder as WorkerBuilder, LazyWorker, Worker},
    PANIC_REGION_FILE_PREFIX,
};
use tokio::runtime::Builder;

use crate::{
    node::*,
    raftkv::*,
    resolve,
    server::Server,
    service::ImportSstService,
    setup::{initial_logger, initial_metric, validate_and_persist_config},
    status_server::StatusServer,
};

const RESERVED_OPEN_FDS: u64 = 1000;

const DEFAULT_METRICS_FLUSH_INTERVAL: Duration = Duration::from_millis(10_000);

const ZSTD_COMPRESSION_LEVEL_FOR_LOCAL: &str = "3";

/// A complete TiKV server.
pub struct TikvServer {
    config: TikvConfig,
    cfg_controller: Option<ConfigController>,
    security_mgr: Arc<SecurityManager>,
    pd_client: Arc<dyn PdClient>,
    system: Option<RaftBatchSystem>,
    router: RaftRouter,
    resolver: resolve::PdStoreAddrResolver,
    store_path: PathBuf,
    raw_engines: Engines,
    engines: Option<TikvEngines>,
    servers: Option<Servers>,
    region_info_accessor: RegionInfoAccessor,
    coprocessor_host: Option<CoprocessorHost<kvengine::Engine>>,
    to_stop: Vec<Box<dyn Stop>>,
    lock_files: Vec<File>,
    concurrency_manager: ConcurrencyManager,
    env: Arc<Environment>,
    background_worker: Worker,
    quota_limiter: Arc<QuotaLimiter>,
    io_rate_limiter: Arc<IoRateLimiter>,
    overload_protector: OverloadProtector,
}

struct TikvEngines {
    store_meta: Option<StoreMeta>,
    engine: RaftKv,
}

struct Servers {
    lock_mgr: LockManager,
    server: Server<RaftRouter, resolve::PdStoreAddrResolver>,
    node: Node,
    importer: Arc<SstImporter>,
}

impl TikvServer {
    pub fn new(mut config: TikvConfig) -> TikvServer {
        let (security_mgr, env, pd, dfs) = Self::prepare(&mut config);
        Self::setup(config, security_mgr, env, pd, dfs)
    }

    #[allow(clippy::type_complexity)]
    pub fn prepare(
        config: &mut TikvConfig,
    ) -> (
        Arc<SecurityManager>,
        Arc<Environment>,
        Arc<dyn PdClient>,
        Arc<dyn Dfs>,
    ) {
        // Sets the global logger ASAP.
        // It is okay to use the config w/o `validate()`,
        // because `initial_logger()` handles various conditions.
        initial_logger(config);

        // Print version information.
        let build_timestamp = option_env!("TIKV_BUILD_TIME");
        tikv::log_tikv_info(build_timestamp);

        // Print resource quota.
        SysQuota::log_quota();
        CPU_CORES_QUOTA_GAUGE.set(SysQuota::cpu_cores_quota());

        // Do some prepare works before start.
        pre_start();

        let _m = Monitor::default();
        tikv_util::thread_group::set_properties(Some(GroupProperties::default()));
        // It is okay use pd config and security config before `init_config`,
        // because these configs must be provided by command line, and only
        // used during startup process.
        let security_mgr = Arc::new(
            SecurityManager::new(&config.security)
                .unwrap_or_else(|e| fatal!("failed to create security manager: {}", e)),
        );
        let env = Arc::new(
            EnvBuilder::new()
                .cq_count(config.server.grpc_concurrency)
                .name_prefix(thd_name!(GRPC_THREAD_PREFIX))
                .build(),
        );
        let pd_client =
            TikvServer::connect_to_pd_cluster(config, env.clone(), Arc::clone(&security_mgr));

        config.dfs.override_from_env();
        config.security.master_key.override_from_env();

        // If zstd_compression_level is not set, set it to default value
        if config.dfs.zstd_compression_level.is_empty() {
            config.dfs.zstd_compression_level = ZSTD_COMPRESSION_LEVEL_FOR_LOCAL.to_string();
        }

        let dfs_conf = &config.dfs;
        let dfs: Arc<dyn Dfs> = if dfs_conf.s3_bucket.is_empty() && dfs_conf.s3_endpoint.is_empty()
            || dfs_conf.s3_endpoint == "local"
        {
            let local_path = PathBuf::from(&config.storage.data_dir).join(Path::new("local"));
            Arc::new(kvengine::dfs::LocalFs::new(&local_path))
        } else if dfs_conf.s3_endpoint == "memory" {
            Arc::new(kvengine::dfs::InMemFs::new())
        } else {
            Arc::new(kvengine::dfs::S3Fs::new(
                dfs_conf.prefix.clone(),
                dfs_conf.s3_endpoint.clone(),
                dfs_conf.s3_key_id.clone(),
                dfs_conf.s3_secret_key.clone(),
                dfs_conf.s3_region.clone(),
                dfs_conf.s3_bucket.clone(),
            ))
        };

        (security_mgr, env, pd_client, dfs)
    }

    pub fn setup(
        config: TikvConfig,
        security_mgr: Arc<SecurityManager>,
        env: Arc<Environment>,
        pd_client: Arc<dyn PdClient>,
        dfs: Arc<dyn Dfs>,
    ) -> TikvServer {
        // Initialize and check config
        let cfg_controller = Self::init_config(config);
        let config = cfg_controller.get_current();
        let io_rate_limiter = Arc::new(IoRateLimiter::new(IoRateLimitMode::WriteOnly, true, true));
        io_rate_limiter
            .set_io_rate_limit(config.storage.io_rate_limit.max_bytes_per_sec.0 as usize);
        let raw_engines = Self::init_raw_engines(
            pd_client.clone(),
            &config,
            dfs,
            io_rate_limiter.clone(),
            security_mgr.clone(),
        );

        let store_path = Path::new(&config.storage.data_dir).to_owned();

        // Initialize raftstore channels.
        let mut rfstore_conf =
            rfstore::store::Config::from_old(&config.raft_store, &config.coprocessor);
        rfstore_conf.enable_inner_key_offset = config.enable_inner_key_offset;
        let system = rfstore::store::RaftBatchSystem::new(&raw_engines, &rfstore_conf);
        let router = system.router();

        let thread_count = config.server.background_thread_count;
        let background_worker = WorkerBuilder::new("background")
            .thread_count(thread_count)
            .create();

        let resolver =
            resolve::new_resolver(Arc::clone(&pd_client), &background_worker, router.clone());
        let mut coprocessor_host = Some(CoprocessorHost::default());
        let region_info_accessor = RegionInfoAccessor::new(coprocessor_host.as_mut().unwrap());

        // Initialize concurrency manager
        let latest_ts = block_on(pd_client.get_tso()).expect("failed to get timestamp from PD");
        let concurrency_manager = ConcurrencyManager::new(latest_ts);

        // use different quota for front-end and back-end requests
        let quota_limiter = Arc::new(QuotaLimiter::new(
            config.quota.foreground_cpu_time,
            config.quota.foreground_write_bandwidth,
            config.quota.foreground_read_bandwidth,
            config.quota.background_cpu_time,
            config.quota.background_write_bandwidth,
            config.quota.background_read_bandwidth,
            config.quota.max_delay_duration,
            config.quota.enable_auto_tune,
        ));
        let mut overload_protector_worker = OverloadProtectorWorker::new(config.overload.clone());
        let overload_protector = overload_protector_worker.get_protector();
        std::thread::spawn(move || {
            overload_protector_worker.run();
        });
        info!("created tikv server");
        TikvServer {
            config,
            cfg_controller: Some(cfg_controller),
            security_mgr,
            pd_client,
            router,
            system: Some(system),
            resolver,
            store_path,
            raw_engines,
            engines: None,
            servers: None,
            region_info_accessor,
            coprocessor_host,
            to_stop: vec![],
            lock_files: vec![],
            concurrency_manager,
            env,
            background_worker,
            quota_limiter,
            io_rate_limiter,
            overload_protector,
        }
    }

    pub fn run(&mut self) {
        let memory_limit = self.config.memory_usage_limit.unwrap().0;
        let high_water = (self.config.memory_usage_high_water * memory_limit as f64) as u64;
        register_memory_usage_high_water(high_water);

        self.check_conflict_addr();
        self.init_fs();
        self.init_yatp();
        self.init_engines();

        let server_config = dispatch_api_version!(self.config.storage.api_version(), {
            self.init_servers::<API>()
        });

        self.register_services();
        let fetcher = self.init_io_utility();
        self.init_metrics_flusher(fetcher);
        self.run_server(server_config);
        self.run_status_server();
        if self.config.gc.enable_safe_point_v2 {
            self.run_watch_ks_gc_safepoint();
        }
    }

    fn run_watch_ks_gc_safepoint(&mut self) {
        let pd_clone = self.pd_client.clone();
        self.background_worker.remote().spawn(async move {
            pd_clone.watch_gc_safepoint_v2().await;
        });
    }

    /// Initialize and check the config
    ///
    /// Warnings are logged and fatal errors exist.
    /// This method is also used by cse-ctl for cluster restore.
    ///
    /// #  Fatal errors
    ///
    /// - If `dynamic config` feature is enabled and failed to register config
    ///   to PD
    /// - If some critical configs (like data dir) are differrent from last run
    /// - If the config can't pass `validate()`
    /// - If the max open file descriptor limit is not high enough to support
    ///   the main database and the raft database.
    pub fn init_config(mut config: TikvConfig) -> ConfigController {
        validate_and_persist_config(&mut config, true);

        ensure_dir_exist(&config.storage.data_dir).unwrap();
        if !config.rocksdb.wal_dir.is_empty() {
            ensure_dir_exist(&config.rocksdb.wal_dir).unwrap();
        }
        if config.raft_engine.enable {
            ensure_dir_exist(&config.raft_engine.config().dir).unwrap();
        } else {
            ensure_dir_exist(&config.raft_store.raftdb_path).unwrap();
            if !config.raftdb.wal_dir.is_empty() {
                ensure_dir_exist(&config.raftdb.wal_dir).unwrap();
            }
        }

        check_system_config(&config);

        tikv_util::set_panic_hook(config.abort_on_panic, &config.storage.data_dir);

        info!(
            "using config";
            "config" => serde_json::to_string(&config).unwrap(),
        );
        if config.panic_when_unexpected_key_or_data {
            info!("panic-when-unexpected-key-or-data is on");
            tikv_util::set_panic_when_unexpected_key_or_data(true);
        }

        config.write_into_metrics();

        ConfigController::new(config)
    }

    fn connect_to_pd_cluster(
        config: &mut TikvConfig,
        env: Arc<Environment>,
        security_mgr: Arc<SecurityManager>,
    ) -> Arc<RpcClient> {
        let pd_client = Arc::new(
            RpcClient::new(&config.pd, Some(env), security_mgr)
                .unwrap_or_else(|e| fatal!("failed to create rpc client: {}", e)),
        );

        let cluster_id = pd_client
            .get_cluster_id()
            .unwrap_or_else(|e| fatal!("failed to get cluster id: {}", e));
        if cluster_id == DEFAULT_CLUSTER_ID {
            fatal!("cluster id can't be {}", DEFAULT_CLUSTER_ID);
        }
        config.server.cluster_id = cluster_id;
        info!(
            "connect to PD cluster";
            "cluster_id" => cluster_id
        );

        pd_client
    }

    fn check_conflict_addr(&mut self) {
        let cur_addr: SocketAddr = self
            .config
            .server
            .addr
            .parse()
            .expect("failed to parse into a socket address");
        let cur_ip = cur_addr.ip();
        let cur_port = cur_addr.port();
        let lock_dir = get_lock_dir();

        let search_base = env::temp_dir().join(lock_dir);
        std::fs::create_dir_all(&search_base)
            .unwrap_or_else(|_| panic!("create {} failed", search_base.display()));

        for entry in fs::read_dir(&search_base).unwrap().flatten() {
            if !entry.file_type().unwrap().is_file() {
                continue;
            }
            let file_path = entry.path();
            let file_name = file_path.file_name().unwrap().to_str().unwrap();
            if let Ok(addr) = file_name.replace('_', ":").parse::<SocketAddr>() {
                let ip = addr.ip();
                let port = addr.port();
                if cur_port == port
                    && (cur_ip == ip || cur_ip.is_unspecified() || ip.is_unspecified())
                {
                    let _ = try_lock_conflict_addr(file_path);
                }
            }
        }

        let cur_path = search_base.join(cur_addr.to_string().replace(':', "_"));
        let cur_file = try_lock_conflict_addr(cur_path);
        self.lock_files.push(cur_file);
    }

    fn init_fs(&mut self) {
        let lock_path = self.store_path.join(Path::new("LOCK"));

        let f = File::create(lock_path.as_path())
            .unwrap_or_else(|e| fatal!("failed to create lock at {}: {}", lock_path.display(), e));
        if f.try_lock_exclusive().is_err() {
            fatal!(
                "lock {} failed, maybe another instance is using this directory.",
                self.store_path.display()
            );
        }
        self.lock_files.push(f);

        if tikv_util::panic_mark_file_exists(&self.config.storage.data_dir) {
            fatal!(
                "panic_mark_file {} exists, there must be something wrong with the db. \
                     Do not remove the panic_mark_file and force the TiKV node to restart. \
                     Please contact TiKV maintainers to investigate the issue. \
                     If needed, use scale in and scale out to replace the TiKV node. \
                     https://docs.pingcap.com/tidb/stable/scale-tidb-using-tiup",
                tikv_util::panic_mark_file_path(&self.config.storage.data_dir).display()
            );
        }
    }

    fn init_yatp(&self) {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            yatp::metrics::set_namespace(Some("tikv"));
            prometheus::register(Box::new(yatp::metrics::MULTILEVEL_LEVEL0_CHANCE.clone()))
                .unwrap();
            prometheus::register(Box::new(yatp::metrics::MULTILEVEL_LEVEL_ELAPSED.clone()))
                .unwrap();
        })
    }

    fn init_engines(&mut self) {
        info!("init engines");
        let store_meta = StoreMeta::new(PENDING_MSG_CAP);
        let engine = RaftKv::new(
            ServerRaftStoreRouter::new(
                self.router.clone(),
                LocalReader::new(
                    self.raw_engines.kv.clone(),
                    store_meta.readers.clone(),
                    self.router.clone(),
                ),
            ),
            self.raw_engines.kv.clone(),
        );
        let store_meta = Some(store_meta);
        self.engines = Some(TikvEngines { store_meta, engine });
    }

    fn init_servers<F: KvFormat>(&mut self) -> Arc<VersionTrack<ServerConfig>> {
        info!("init servers");

        let cfg_controller = self.cfg_controller.as_mut().unwrap();

        cfg_controller.register(
            tikv::config::Module::Quota,
            Box::new(QuotaLimitConfigManager::new(Arc::clone(
                &self.quota_limiter,
            ))),
        );

        let lock_mgr = LockManager::new(&self.config.pessimistic_txn);
        lock_mgr.register_detector_role_change_observer(self.coprocessor_host.as_mut().unwrap());

        let engines = self.engines.as_mut().unwrap();

        let pd_worker = LazyWorker::new("pd-worker");
        let pd_sender = pd_worker.scheduler();
        let flow_reporter = rfstore::store::worker::FlowStatsReporter::new(pd_sender.clone());

        let unified_read_pool = if self.config.readpool.is_unified_pool_enabled() {
            Some(build_yatp_read_pool(
                &self.config.readpool.unified,
                flow_reporter.clone(),
                engines.engine.clone(),
            ))
        } else {
            None
        };

        // The `DebugService` and `DiagnosticsService` will share the same thread pool
        let props = tikv_util::thread_group::current_properties();
        let debug_thread_pool = Arc::new(
            Builder::new_multi_thread()
                .thread_name(thd_name!("debugger"))
                .worker_threads(1)
                .after_start_wrapper(move || {
                    tikv_alloc::add_thread_memory_accessor();
                    tikv_util::thread_group::set_properties(props.clone());
                })
                .before_stop_wrapper(tikv_alloc::remove_thread_memory_accessor)
                .build()
                .unwrap(),
        );
        // Start resource metering.
        let (recorder_notifier, collector_reg_handle, resource_tag_factory, recorder_worker) =
            resource_metering::init_recorder(self.config.resource_metering.precision.as_millis());
        self.to_stop.push(recorder_worker);
        let (reporter_notifier, data_sink_reg_handle, reporter_worker) =
            resource_metering::init_reporter(
                self.config.resource_metering.clone(),
                collector_reg_handle,
            );
        self.to_stop.push(reporter_worker);
        let (address_change_notifier, single_target_worker) = resource_metering::init_single_target(
            self.config.resource_metering.receiver_address.clone(),
            self.env.clone(),
            data_sink_reg_handle,
        );
        self.to_stop.push(single_target_worker);

        let cfg_manager = resource_metering::ConfigManager::new(
            self.config.resource_metering.clone(),
            recorder_notifier,
            reporter_notifier,
            address_change_notifier,
        );
        cfg_controller.register(
            tikv::config::Module::ResourceMetering,
            Box::new(cfg_manager),
        );

        let overload_cfg_manager = overload_protector::OverloadConfigManager::new(
            self.overload_protector.clone(),
            self.config.overload.clone(),
        );
        cfg_controller.register(
            tikv::config::Module::Overload,
            Box::new(overload_cfg_manager),
        );

        let storage_read_pool_handle = if self.config.readpool.storage.use_unified_pool() {
            unified_read_pool.as_ref().unwrap().handle()
        } else {
            let storage_read_pools = ReadPool::from(storage::build_read_pool(
                &self.config.readpool.storage,
                flow_reporter.clone(),
                engines.engine.clone(),
            ));
            storage_read_pools.handle()
        };
        let reporter = rfstore::store::FlowStatsReporter::new(pd_sender);
        let storage = create_raft_storage::<_, F>(
            engines.engine.clone(),
            &self.config.storage,
            storage_read_pool_handle,
            lock_mgr.clone(),
            self.concurrency_manager.clone(),
            lock_mgr.get_storage_dynamic_configs(),
            Arc::new(FlowController::Singleton(EngineFlowController::empty())),
            reporter,
            resource_tag_factory.clone(),
            Arc::clone(&self.quota_limiter),
            self.pd_client.feature_gate().clone(),
        )
        .unwrap_or_else(|e| fatal!("failed to create raft storage: {}", e));

        ReplicaReadLockChecker::new(self.concurrency_manager.clone())
            .register(self.coprocessor_host.as_mut().unwrap());

        // Create coprocessor endpoint.
        let cop_read_pool_handle = if self.config.readpool.coprocessor.use_unified_pool() {
            unified_read_pool.as_ref().unwrap().handle()
        } else {
            let cop_read_pools = ReadPool::from(coprocessor::readpool_impl::build_read_pool(
                &self.config.readpool.coprocessor,
                flow_reporter,
                engines.engine.clone(),
            ));
            cop_read_pools.handle()
        };

        let server_config = Arc::new(VersionTrack::new(self.config.server.clone()));

        self.config
            .raft_store
            .validate(
                self.config.coprocessor.region_split_size,
                self.config.coprocessor.enable_region_bucket,
                self.config.coprocessor.region_bucket_size,
            )
            .unwrap_or_else(|e| fatal!("failed to validate raftstore config {}", e));
        let mut raftstore_conf =
            rfstore::store::Config::from_old(&self.config.raft_store, &self.config.coprocessor);
        raftstore_conf.enable_inner_key_offset = self.config.enable_inner_key_offset;
        let raft_store = Arc::new(VersionTrack::new(raftstore_conf));
        let mut node = Node::new(
            self.system.take().unwrap(),
            &server_config.value().clone(),
            raft_store,
            self.pd_client.clone(),
            self.background_worker.clone(),
        );
        info!("bootstrap store");
        node.try_bootstrap_store(self.raw_engines.clone())
            .unwrap_or_else(|e| fatal!("failed to bootstrap node id: {}", e));
        info!("store bootstrapped");

        let mut copr = coprocessor::Endpoint::new(
            &server_config.value(),
            cop_read_pool_handle,
            self.concurrency_manager.clone(),
            resource_tag_factory,
            Arc::new(QuotaLimiter::default()),
            Some(self.overload_protector.clone()),
            self.security_mgr.clone(),
        );
        copr.set_remote_url(
            self.config.dfs.remote_analyzer_addr.clone(),
            self.config.kvengine.remote_coprocessor_addr.clone(),
            self.config.kvengine.remote_coprocessor_min_blocks,
        );
        // Create server
        let server = Server::new(
            node.id(),
            &server_config,
            &self.security_mgr,
            storage,
            copr,
            coprocessor_v2::Endpoint::new(&self.config.coprocessor_v2),
            self.router.clone(),
            self.resolver.clone(),
            self.env.clone(),
            unified_read_pool,
            debug_thread_pool,
        )
        .unwrap_or_else(|e| fatal!("failed to create server: {}", e));

        let import_path = self.store_path.join("import");
        let mut importer = SstImporter::new(
            &self.config.import,
            import_path,
            None,
            self.config.storage.api_version(),
        )
        .unwrap();
        for (cf_name, compression_type) in &[
            (
                CF_DEFAULT,
                self.config.rocksdb.defaultcf.bottommost_level_compression,
            ),
            (
                CF_WRITE,
                self.config.rocksdb.writecf.bottommost_level_compression,
            ),
        ] {
            importer.set_compression_type(cf_name, from_rocks_compression_type(*compression_type));
        }
        let importer = Arc::new(importer);

        // `ConsistencyCheckObserver` must be registered before `Node::start`.
        let safe_point = Arc::new(AtomicU64::new(0));
        let observer = match self.config.coprocessor.consistency_check_method {
            ConsistencyCheckMethod::Mvcc => {
                BoxConsistencyCheckObserver::new(MvccConsistencyCheckObserver::new(safe_point))
            }
            ConsistencyCheckMethod::Raw => {
                BoxConsistencyCheckObserver::new(RawConsistencyCheckObserver::default())
            }
        };
        self.coprocessor_host
            .as_mut()
            .unwrap()
            .registry
            .register_consistency_check_observer(100, observer);

        node.start(
            self.raw_engines.clone(),
            Box::new(server.transport()),
            pd_worker,
            engines.store_meta.take().unwrap(),
            self.coprocessor_host.clone().unwrap(),
            importer.clone(),
            self.concurrency_manager.clone(),
        )
        .unwrap_or_else(|e| panic!("failed to start node: {:?}", e));

        initial_metric(&self.config.metric);

        self.servers = Some(Servers {
            lock_mgr,
            server,
            node,
            importer,
        });

        server_config
    }

    fn register_services(&mut self) {
        let servers = self.servers.as_mut().unwrap();
        let engines = self.engines.as_ref().unwrap();

        // Import SST service.
        let import_service = ImportSstService::new(
            self.config.import.clone(),
            self.config.raft_store.raft_entry_max_size,
            self.router.clone(),
            engines.engine.kv_engine().unwrap(),
            servers.importer.clone(),
        );
        if servers
            .server
            .register_service(create_import_sst(import_service))
            .is_some()
        {
            fatal!("failed to register import service");
        }

        // Lock manager.
        if servers
            .server
            .register_service(create_deadlock(servers.lock_mgr.deadlock_service()))
            .is_some()
        {
            fatal!("failed to register deadlock service");
        }

        servers
            .lock_mgr
            .start(
                servers.node.id(),
                self.pd_client.clone(),
                self.resolver.clone(),
                self.security_mgr.clone(),
                &self.config.pessimistic_txn,
            )
            .unwrap_or_else(|e| fatal!("failed to start lock manager: {}", e));

        // Backup service.
        let mut backup_worker = Box::new(self.background_worker.lazy_build("backup-endpoint"));
        let backup_scheduler = backup_worker.scheduler();
        let backup_service = backup::Service::<kvengine::Engine>::new(backup_scheduler);
        if servers
            .server
            .register_service(create_backup(backup_service))
            .is_some()
        {
            fatal!("failed to register backup service");
        }

        let backup_endpoint = backup::Endpoint::new(
            servers.node.id(),
            engines.engine.clone(),
            self.region_info_accessor.clone(),
            engines.engine.kv_engine().unwrap(),
            self.config.backup.clone(),
            self.concurrency_manager.clone(),
            self.config.storage.api_version(),
            None,
        );
        self.cfg_controller.as_mut().unwrap().register(
            tikv::config::Module::Backup,
            Box::new(backup_endpoint.get_config_manager()),
        );
        backup_worker.start(backup_endpoint);
    }

    fn init_io_utility(&mut self) -> BytesFetcher {
        let stats_collector_enabled = file_system::init_io_stats_collector()
            .map_err(|e| warn!("failed to init I/O stats collector: {}", e))
            .is_ok();

        if stats_collector_enabled {
            BytesFetcher::FromIoStatsCollector()
        } else {
            BytesFetcher::FromRateLimiter(self.io_rate_limiter.statistics().unwrap())
        }
    }

    fn init_metrics_flusher(&mut self, fetcher: BytesFetcher) {
        let mut io_metrics = IoMetricsManager::new(fetcher);
        let kv = self.raw_engines.kv.clone();
        let raft = self.raw_engines.raft.clone();
        self.background_worker
            .spawn_interval_task(DEFAULT_METRICS_FLUSH_INTERVAL, move || {
                let now = Instant::now();
                KvEngine::flush_metrics(&kv, "kv");
                RaftEngine::flush_metrics(&raft, "raft");
                io_metrics.flush(now);
            });
    }

    fn run_server(&mut self, server_config: Arc<VersionTrack<ServerConfig>>) {
        let server = self.servers.as_mut().unwrap();
        server
            .server
            .build_and_bind()
            .unwrap_or_else(|e| fatal!("failed to build server: {}", e));
        server
            .server
            .start(server_config, self.security_mgr.clone())
            .unwrap_or_else(|e| fatal!("failed to start server: {}", e));
    }

    fn run_status_server(&mut self) {
        // Create a status server.
        let status_enabled = !self.config.server.status_addr.is_empty();
        if status_enabled {
            let mut status_server = match StatusServer::new(
                self.config.server.status_thread_pool_size,
                self.cfg_controller.take().unwrap(),
                Arc::new(self.config.security.clone()),
                self.router.clone(),
                self.store_path.clone(),
                self.raw_engines.kv.clone(),
                self.raw_engines.raft.clone(),
            ) {
                Ok(status_server) => Box::new(status_server),
                Err(e) => {
                    error_unknown!(%e; "failed to start runtime for status service");
                    return;
                }
            };
            // Start the status server.
            if let Err(e) = status_server.start(self.config.server.status_addr.clone()) {
                error_unknown!(%e; "failed to bind addr for status service");
            } else {
                self.to_stop.push(status_server);
            }
        }
    }

    pub fn stop(self) {
        self.force_stop(false);
    }

    pub fn force_stop(self, force: bool) {
        tikv_util::thread_group::mark_shutdown();
        let mut servers = self.servers.unwrap();
        servers
            .server
            .stop()
            .unwrap_or_else(|e| fatal!("failed to stop server: {}", e));

        servers.node.stop();
        self.region_info_accessor.stop();

        servers.lock_mgr.stop();

        self.to_stop.into_iter().for_each(|s| s.stop());
        self.raw_engines.raft.stop_worker(force);
        self.overload_protector.stop();
        self.background_worker.stop();
    }

    pub fn get_kv_engine(&self) -> kvengine::Engine {
        self.raw_engines.kv.clone()
    }

    pub fn get_raft_engine(&self) -> rfengine::RfEngine {
        self.raw_engines.raft.clone()
    }

    pub fn get_store_id(&self) -> u64 {
        self.servers.as_ref().unwrap().node.id()
    }

    pub fn get_sst_importer(&self) -> Arc<SstImporter> {
        self.servers.as_ref().unwrap().importer.clone()
    }

    pub fn get_raft_router(&self) -> RaftRouter {
        self.router.clone()
    }
}

impl TikvServer {
    // This method is also used by cse-ctl for cluster restore.
    pub fn init_raft_engine(conf: &TikvConfig) -> rfengine::Result<RfEngine> {
        let raft_db_path = Path::new(&conf.raft_store.raftdb_path);
        let data_dir = Path::new(&conf.storage.data_dir);
        RfEngine::open(
            raft_db_path,
            &conf.rfengine,
            Some(data_dir),
            Some(conf.dfs.clone()),
        )
    }

    // This method is also used by cse-ctl for cluster restore.
    pub fn init_kv_engine(
        pd: Arc<dyn PdClient>,
        conf: &TikvConfig,
        dfs: Arc<dyn Dfs>,
        rate_limiter: Arc<IoRateLimiter>,
        meta_iter: &mut impl kvengine::MetaIterator,
        recoverer: impl kvengine::RecoverHandler + 'static,
        for_restore: bool,
        security_mgr: Arc<SecurityManager>,
    ) -> kvengine::Result<(
        kvengine::Engine,
        mpsc::Sender<StoreMsg>,
        mpsc::Receiver<StoreMsg>,
    )> {
        let kv_engine_path = PathBuf::from(&conf.storage.data_dir).join(Path::new("db"));
        let mut kv_opts = kvengine::Options::default();
        let capacity = match conf.storage.block_cache.capacity {
            None => {
                let total_mem = SysQuota::memory_limit_in_bytes();
                ((total_mem as f64) * tikv::config::BLOCK_CACHE_RATE) as usize
            }
            Some(c) => c.0 as usize,
        };
        kv_opts.local_dir = kv_engine_path;
        kv_opts.num_compactors = conf.rocksdb.max_background_jobs as usize;
        kv_opts.max_mem_table_size = conf.rocksdb.writecf.write_buffer_size.0;
        if kv_opts.max_mem_table_size > kvengine::KV_ENGINE_MEM_TABLE_MAX_SIZE {
            fatal!(
                "max_mem_table_size {} is too large, should be no more than {}",
                kv_opts.max_mem_table_size,
                kvengine::KV_ENGINE_MEM_TABLE_MAX_SIZE
            );
        }
        // base_size affects compaction priority a lot, we should cap it to a smaller
        // size when we increase the region_split_size.
        kv_opts.base_size = (conf.coprocessor.region_split_size.0 / 16).min(32 * 1024 * 1024);
        kv_opts.max_block_cache_size = capacity as i64;
        kv_opts.remote_compactor_addr = conf.dfs.remote_compactor_addr.clone();
        kv_opts.enable_safe_point_v2 = conf.gc.enable_safe_point_v2;
        kv_opts.disable_safe_point_fallback_v1 = conf.gc.disable_safe_point_fallback_v1;
        let cf_opt = &conf.rocksdb.writecf;
        kv_opts.table_builder_options.block_size = cf_opt.block_size.0 as usize;
        kv_opts.table_builder_options.max_table_size = cf_opt.target_file_size_base.0 as usize;
        kv_opts.table_builder_options.compression_lvl =
            conf.dfs.zstd_compression_level.parse().unwrap_or_else(|_| {
                fatal!(
                    "invalid zstd compression level: {}",
                    conf.dfs.zstd_compression_level
                )
            });
        kv_opts.allow_fallback_local = conf.dfs.allow_fallback_local;
        kv_opts.enable_inner_key_offset = conf.enable_inner_key_offset;
        kv_opts.max_del_range_delay = conf.kvengine.max_del_range_delay.into();
        kv_opts.compaction_request_version = conf.kvengine.compaction_request_version;
        kv_opts.compaction_tombs_ratio = conf.kvengine.compaction_tombs_ratio;
        kv_opts.compaction_tombs_count = conf.kvengine.compaction_tombs_count;
        kv_opts.for_restore = for_restore;
        let opts = Arc::new(kv_opts);
        let id_allocator = Arc::new(PdIdAllocator::new(pd.clone()));

        let (sender, receiver) = tikv_util::mpsc::unbounded();
        let meta_change_listener = Box::new(MetaChangeListener {
            sender: sender.clone(),
        });

        let mut opt_ks_gc_sp_cache = None;
        if conf.gc.enable_safe_point_v2 {
            opt_ks_gc_sp_cache = Some(pd.get_keyspace_gc_safepoint_v2_cache());
        }
        let master_key = dfs.get_runtime().block_on(conf.security.new_master_key());
        let kv_engine = kvengine::Engine::open(
            dfs,
            opts,
            conf.kvengine.clone(),
            meta_iter,
            recoverer,
            id_allocator,
            meta_change_listener,
            rate_limiter,
            opt_ks_gc_sp_cache,
            master_key,
            security_mgr,
        )?;
        Ok((kv_engine, sender, receiver))
    }

    fn init_raw_engines(
        pd: Arc<dyn PdClient>,
        conf: &TikvConfig,
        dfs: Arc<dyn Dfs>,
        rate_limiter: Arc<IoRateLimiter>,
        security_mgr: Arc<SecurityManager>,
    ) -> Engines {
        let panic_regions = Self::load_panic_regions(&conf.storage.data_dir);
        let black_list_regions = panic_regions
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(id, _)| id)
            .collect();
        let rf_engine = Self::init_raft_engine(conf).unwrap();
        let recoverer = rfstore::store::RecoverHandler::new(rf_engine.clone());
        let mut meta_iter = recoverer.clone();
        if let Some(mut black_list) = load_black_list(&conf.black_list_path) {
            black_list.add_regions(black_list_regions);
            meta_iter.set_black_list(black_list);
        } else if !black_list_regions.is_empty() {
            let black_list = BlackList::new(vec![], black_list_regions);
            meta_iter.set_black_list(black_list);
        }
        if let Some(region_ids) = get_store_regions(pd.clone(), conf, rf_engine.clone()) {
            meta_iter.set_contained_region_ids(region_ids);
        }
        let (kv_engine, sender, receiver) = Self::init_kv_engine(
            pd,
            conf,
            dfs,
            rate_limiter,
            &mut meta_iter,
            recoverer,
            false,
            security_mgr,
        )
        .unwrap();
        Engines::new(
            kv_engine,
            rf_engine,
            (sender, receiver),
            meta_iter.take_black_list(),
        )
    }

    fn load_panic_regions<P: AsRef<Path>>(data_dir: P) -> Vec<(u64, usize)> {
        let dir = fs::read_dir(data_dir).unwrap();
        let mut panic_regions = vec![];
        for entry in dir.into_iter().flatten() {
            let file_name = entry.file_name().into_string().unwrap_or_default();
            if let Some(region_id_str) = file_name.strip_prefix(PANIC_REGION_FILE_PREFIX) {
                if let Ok(region_id) = region_id_str.parse::<u64>() {
                    let count = get_panic_region_count(entry.path()) as usize;
                    panic_regions.push((region_id, count));
                }
            }
        }
        panic_regions
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug, Default)]
#[serde(default)]
struct BlackListConfig {
    keyspace_ids: Vec<u32>,
    region_ids: Vec<u64>,
}

fn load_black_list(black_list_path: &str) -> Option<BlackList> {
    if black_list_path.is_empty() {
        return None;
    }
    if let Ok(data) = fs::read(black_list_path) {
        match serde_json::from_slice::<BlackListConfig>(&data) {
            Ok(config) => return Some(BlackList::new(config.keyspace_ids, config.region_ids)),
            Err(err) => {
                error!("failed to load black list file {:?}", err);
            }
        }
    }
    None
}

fn get_store_regions(
    pd: Arc<dyn pd_client::PdClient>,
    conf: &TikvConfig,
    rf_engine: RfEngine,
) -> Option<Vec<u64>> {
    if !pd.is_cluster_bootstrapped().unwrap() {
        return None;
    };
    let store_id = match rf_engine.get_state(0, STORE_IDENT_KEY) {
        None => 0,
        Some(bin) => {
            let mut ident = StoreIdent::default();
            ident.merge_from_bytes(&bin).unwrap();
            ident.get_store_id()
        }
    };
    if store_id == INVALID_ID {
        return None;
    }
    info!("store_id: {}", store_id);
    let elapsed_secs = match pd.get_store(store_id) {
        Ok(store) => {
            let last_heartbeat = store.get_last_heartbeat();
            if last_heartbeat > 0 {
                let last_heartbeat_dur = Duration::from_nanos(last_heartbeat as u64);
                chrono::Local::now().timestamp() as u64 - last_heartbeat_dur.as_secs()
            } else {
                return None;
            }
        }
        Err(e) => {
            warn!("failed to get store: {}", e);
            return None;
        }
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let pd_control = PdControl::new(conf.pd.clone(), pd.get_security_mgr()).unwrap();
    let max_store_down_time_secs = match rt.block_on(pd_control.get_config()) {
        Ok(pd_config) => {
            if !pd_config.schedule.max_store_down_time.is_empty() {
                ReadableDuration::from_str(pd_config.schedule.max_store_down_time.as_str())
                    .unwrap()
                    .as_secs()
            } else {
                return None;
            }
        }
        Err(e) => {
            warn!("failed to get config: {}", e);
            return None;
        }
    };
    info!(
        "elapsed secs: {}, max store down time secs: {}",
        elapsed_secs, max_store_down_time_secs
    );
    if max_store_down_time_secs > 0 && elapsed_secs > max_store_down_time_secs {
        match rt.block_on(pd_control.get_store_regions(store_id)) {
            Ok(regions_info) => {
                info!("get store regions: {}", regions_info.regions.len());
                let region_ids = regions_info
                    .regions
                    .into_iter()
                    .map(|region| region.id)
                    .collect();
                return Some(region_ids);
            }
            Err(e) => {
                warn!("failed to get store regions: {}", e);
            }
        };
    }
    None
}

/// Various sanity-checks and logging before running a server.
///
/// Warnings are logged.
///
/// # Logs
///
/// The presence of these environment variables that affect the database
/// behavior is logged.
///
/// - `GRPC_POLL_STRATEGY`
/// - `http_proxy` and `https_proxy`
///
/// # Warnings
///
/// - if `net.core.somaxconn` < 32768
/// - if `net.ipv4.tcp_syncookies` is not 0
/// - if `vm.swappiness` is not 0
/// - if data directories are not on SSDs
/// - if the "TZ" environment variable is not set on unix
fn pre_start() {
    check_environment_variables();
    for e in tikv_util::config::check_kernel() {
        warn!(
            "check: kernel";
            "err" => %e
        );
    }
}

fn check_system_config(config: &TikvConfig) {
    info!("beginning system configuration check");
    let mut rocksdb_max_open_files = config.rocksdb.max_open_files;
    if config.rocksdb.titan.enabled {
        // Titan engine maintains yet another pool of blob files and uses the same max
        // number of open files setup as rocksdb does. So we double the max required
        // open files here
        rocksdb_max_open_files *= 2;
    }
    if let Err(e) = tikv_util::config::check_max_open_fds(
        RESERVED_OPEN_FDS + (rocksdb_max_open_files + config.raftdb.max_open_files) as u64,
    ) {
        fatal!("{}", e);
    }

    // Check RocksDB data dir
    if let Err(e) = tikv_util::config::check_data_dir(&config.storage.data_dir) {
        warn!(
            "check: rocksdb-data-dir";
            "path" => &config.storage.data_dir,
            "err" => %e
        );
    }
    // Check raft data dir
    if let Err(e) = tikv_util::config::check_data_dir(&config.raft_store.raftdb_path) {
        warn!(
            "check: raftdb-path";
            "path" => &config.raft_store.raftdb_path,
            "err" => %e
        );
    }
}

fn try_lock_conflict_addr<P: AsRef<Path>>(path: P) -> File {
    let f = File::create(path.as_ref()).unwrap_or_else(|e| {
        fatal!(
            "failed to create lock at {}: {}",
            path.as_ref().display(),
            e
        )
    });

    if f.try_lock_exclusive().is_err() {
        fatal!(
            "{} already in use, maybe another instance is binding with this address.",
            path.as_ref().file_name().unwrap().to_str().unwrap()
        );
    }
    f
}

#[cfg(unix)]
fn get_lock_dir() -> String {
    format!("{}_TIKV_LOCK_FILES", unsafe { libc::getuid() })
}

#[cfg(not(unix))]
fn get_lock_dir() -> String {
    "TIKV_LOCK_FILES".to_owned()
}

/// A small trait for components which can be trivially stopped. Lets us keep
/// a list of these in `TiKV`, rather than storing each component individually.
trait Stop {
    fn stop(self: Box<Self>);
}

impl Stop for StatusServer {
    fn stop(self: Box<Self>) {
        (*self).stop()
    }
}

impl Stop for Worker {
    fn stop(self: Box<Self>) {
        Worker::stop(&self);
    }
}

impl<T: fmt::Display + Send + 'static> Stop for LazyWorker<T> {
    fn stop(self: Box<Self>) {
        self.stop_worker();
    }
}
