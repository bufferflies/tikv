// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::raft_serverpb::RaftMessage;

use super::*;
use crate::{store::Transport, Result};

#[derive(Clone)]
struct TestNotifier {
    tx: Sender<(u64, (u64, u64))>,
}

impl PersistedNotifier for TestNotifier {
    fn notify(&self, region_id: u64, peer_id: u64, ready_number: u64) {
        self.tx.send((region_id, (peer_id, ready_number))).unwrap()
    }
}

#[derive(Clone)]
struct TestTransport {
    tx: Sender<RaftMessage>,
}

impl Transport for TestTransport {
    fn send(&mut self, msg: RaftMessage) -> Result<()> {
        self.tx.send(msg).unwrap();
        Ok(())
    }
    fn set_store_allowlist(&mut self, _: Vec<u64>) {
        unimplemented!();
    }
    fn need_flush(&self) -> bool {
        false
    }
    fn flush(&mut self) {}
}
