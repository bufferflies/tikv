// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

pub use self::imp::wait_for_signal;

#[cfg(unix)]
mod imp {
    use signal_hook::{
        consts::{SIGHUP, SIGINT, SIGTERM, SIGUSR1, SIGUSR2},
        iterator::Signals,
    };

    #[allow(dead_code)]
    pub fn wait_for_signal() {
        let mut signals = Signals::new([SIGTERM, SIGINT, SIGHUP, SIGUSR1, SIGUSR2]).unwrap();
        for signal in &mut signals {
            match signal {
                SIGTERM | SIGINT | SIGHUP => {
                    info!("receive signal {}, stopping server...", signal);
                    break;
                }
                SIGUSR1 => {
                    // Use SIGUSR1 to log metrics.
                    // TODO(x)
                    // info!("{}", metrics::dump());
                    // if let Some(ref engines) = engines {
                    //     info!("{:?}", MiscExt::dump_stats(&engines.kv));
                    //     info!("{:?}", RaftEngine::dump_stats(&engines.raft));
                    // }
                }
                // TODO: handle more signal
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(not(unix))]
mod imp {
    use engine_rocks::RocksEngine;
    use engine_traits::Engines;

    pub fn wait_for_signal(_: Option<Engines<RocksEngine, RocksEngine>>) {}
}
