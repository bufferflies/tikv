// Copyright 2023 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::atomic::Ordering, time::Duration};

use test_cloud_server::keyspace::ClusterKeyspaceClient;
use tikv_util::{info, time::Instant};

use crate::{DROP_TABLE_COUNTER, TABLE_COUNTER};

pub fn spawn_drop_table(
    mut client: ClusterKeyspaceClient,
    interval: Duration,
    timeout: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let start_time = Instant::now_coarse();
        while start_time.saturating_elapsed() < timeout {
            let loop_start_time = Instant::now_coarse();
            let random = || {
                let mut rng = rand::thread_rng();
                let keyspace_id = client.keyspace_manager().get_zipf_random_keyspace(&mut rng);
                let table_id = client
                    .keyspace_manager()
                    .get_random_available_table(keyspace_id, &mut rng);
                (keyspace_id, table_id)
            };
            let (keyspace_id, table_id) = random();
            let table_id = match table_id {
                Some(table_id) => table_id,
                None => continue,
            };

            client.drop_table(keyspace_id, table_id).await.unwrap();
            DROP_TABLE_COUNTER.fetch_add(1, Ordering::Relaxed);

            // Add a new table to keep number of tables.
            client
                .keyspace_manager()
                .get_keyspace_meta(keyspace_id)
                .unwrap()
                .new_table(true);
            TABLE_COUNTER.fetch_add(1, Ordering::Relaxed);

            let sleep_time = interval.saturating_sub(loop_start_time.saturating_elapsed());
            tokio::time::sleep(sleep_time).await;
        }
        info!("drop table thread exit");
    })
}

pub(crate) fn check_drop_table() {
    let counter = DROP_TABLE_COUNTER.load(Ordering::SeqCst);
    assert!(counter >= 3, "too few drop table: {}", counter);
}
