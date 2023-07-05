// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

pub mod client;
pub mod cluster;
pub mod keyspace;
pub mod oss;
pub mod scheduler;
mod txnlock;

pub use cluster::*;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tikv_util::info;

    use crate::ServerCluster;

    #[test]
    fn it_works() {
        test_util::init_log_for_test();
        let mut cluster = ServerCluster::new(vec![1, 2, 3], |_, _| {});
        let stores = cluster.get_stores();
        assert_eq!(stores.len(), 3);
        let mut client = cluster.new_client();
        client.put_kv(0..100, gen_key, gen_val);
        client.put_kv(100..200, gen_key, gen_val);
        client.put_kv(200..300, gen_key, gen_val);
        let split_keys = vec![gen_key(50), gen_key(150), gen_key(200)];
        for split_key in &split_keys {
            client.split(split_key);
        }
        cluster.wait_pd_region_count(4);
        cluster.get_pd_client().disable_default_operator();
        cluster.remove_node_peers(1);
        cluster.stop_node(1);
        std::thread::sleep(Duration::from_millis(100));
        cluster.start_node(1, |_, _| {});
        cluster.get_pd_client().enable_default_operator();
        info!("enable replica operator");
        cluster.wait_region_replicated(&[], 3);
        for split_key in &split_keys {
            cluster.wait_region_replicated(split_key, 3);
        }
        client.verify_data_with_ref_store();
        cluster.stop();
    }

    fn gen_key(i: usize) -> Vec<u8> {
        format!("xkey{:04}", i).into_bytes()
    }

    fn gen_val(i: usize) -> Vec<u8> {
        // `repeat` must > 0. CSE treat empty value as not found.
        format!("val{:04}", i).repeat(i % 32 + 1).into_bytes()
    }
}
