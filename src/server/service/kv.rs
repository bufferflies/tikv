// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

use kvproto::tikvpb::*;
use tikv_alloc::trace::MemoryTraceGuard;
use tikv_util::time::Instant;

use crate::server::metrics::*;

const GRPC_MSG_MAX_BATCH_SIZE: usize = 128;

pub mod batch_commands_response {
    pub type Response = kvproto::tikvpb::BatchCommandsResponseResponse;

    pub mod response {
        pub type Cmd = kvproto::tikvpb::BatchCommandsResponse_Response_oneof_cmd;
    }
}

pub mod batch_commands_request {
    pub type Request = kvproto::tikvpb::BatchCommandsRequestRequest;

    pub mod request {
        pub type Cmd = kvproto::tikvpb::BatchCommandsRequest_Request_oneof_cmd;
    }
}

/// To measure execute time for a given request.
#[derive(Debug)]
pub struct GrpcRequestDuration {
    pub begin: Instant,
    pub label: GrpcTypeKind,
    pub source: String,
}
impl GrpcRequestDuration {
    pub fn new(begin: Instant, label: GrpcTypeKind, source: String) -> Self {
        GrpcRequestDuration {
            begin,
            label,
            source,
        }
    }
}

/// A single response, will be collected into `MeasuredBatchResponse`.
#[derive(Debug)]
pub struct MeasuredSingleResponse {
    pub id: u64,
    pub resp: MemoryTraceGuard<batch_commands_response::Response>,
    pub measure: GrpcRequestDuration,
}
impl MeasuredSingleResponse {
    pub fn new<T>(id: u64, resp: T, measure: GrpcRequestDuration) -> Self
    where
        MemoryTraceGuard<batch_commands_response::Response>: From<T>,
    {
        let resp = resp.into();
        MeasuredSingleResponse { id, resp, measure }
    }
}

/// A batch response.
pub struct MeasuredBatchResponse {
    pub batch_resp: BatchCommandsResponse,
    pub measures: Vec<GrpcRequestDuration>,
}
impl Default for MeasuredBatchResponse {
    fn default() -> Self {
        MeasuredBatchResponse {
            batch_resp: Default::default(),
            measures: Vec::with_capacity(GRPC_MSG_MAX_BATCH_SIZE),
        }
    }
}
