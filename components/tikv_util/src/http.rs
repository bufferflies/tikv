// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use http::header;

pub const CONTENT_TYPE_PROTOBUF: &str = "application/protobuf";

pub trait HeaderExt {
    fn is_accept_protobuf(&self) -> bool;
    fn is_content_type_protobuf(&self) -> bool;
}

impl HeaderExt for http::HeaderMap {
    fn is_accept_protobuf(&self) -> bool {
        self.get_all(header::ACCEPT)
            .iter()
            .any(|v| v == CONTENT_TYPE_PROTOBUF)
    }

    fn is_content_type_protobuf(&self) -> bool {
        self.get(header::CONTENT_TYPE)
            .is_some_and(|x| x == CONTENT_TYPE_PROTOBUF)
    }
}
