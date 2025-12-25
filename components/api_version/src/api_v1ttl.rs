// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use super::*;

impl KvFormat for ApiV1Ttl {
    const TAG: ApiVersion = ApiVersion::V1ttl;
    #[cfg(any(test, feature = "testexport"))]
    const CLIENT_TAG: ApiVersion = ApiVersion::V1;
    const IS_TTL_ENABLED: bool = true;

    #[inline]
    fn parse_key_mode(_: &[u8]) -> KeyMode {
        // In V1TTL, txnkv is disabled, so all data keys are raw keys.
        KeyMode::Raw
    }

    fn parse_range_mode(_: (Option<&[u8]>, Option<&[u8]>)) -> KeyMode {
        // In V1TTL, txnkv is disabled, so all data keys are raw keys.
        KeyMode::Raw
    }
}
