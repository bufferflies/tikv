// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use serde_derive::Deserialize;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum TiCdcError {
    #[error("Server is not ready: {0}")]
    ServerIsNotReady(String),
    #[error("Changefeed already exists: {0}")]
    ChangeFeedAlreadyExists(String),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("{error_msg}")]
    OtherError {
        error_code: String,
        error_msg: String,
    },
}

#[derive(Deserialize, Clone, Debug, Default)]
pub(crate) struct ErrorResponse {
    pub error_msg: String,
    pub error_code: String,
}

pub(crate) fn parse_ticdc_response(resp: &[u8]) -> TiCdcError {
    let Ok(js_resp) = serde_json::from_slice::<ErrorResponse>(resp) else {
        return TiCdcError::InvalidFormat(String::from_utf8_lossy(resp).to_string());
    };

    match js_resp.error_code.as_str() {
        "CDC:ErrChangeFeedAlreadyExists" => TiCdcError::ChangeFeedAlreadyExists(js_resp.error_msg),
        "CDC:ErrServerIsNotReady" => TiCdcError::ServerIsNotReady(js_resp.error_msg),
        _ => TiCdcError::OtherError {
            error_code: js_resp.error_code,
            error_msg: js_resp.error_msg,
        },
    }
}
