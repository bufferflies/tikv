// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

use super::*;

impl KvFormat for ApiV1 {
    const TAG: ApiVersion = ApiVersion::V1;
    #[cfg(any(test, feature = "testexport"))]
    const CLIENT_TAG: ApiVersion = ApiVersion::V1;
    const IS_TTL_ENABLED: bool = false;

    #[inline]
    fn parse_key_mode(_: &[u8]) -> KeyMode {
        KeyMode::Unknown
    }

    fn parse_range_mode(_: (Option<&[u8]>, Option<&[u8]>)) -> KeyMode {
        KeyMode::Unknown
    }
}
