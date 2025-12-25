// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

#![feature(min_specialization)]

mod api_v1;
mod api_v1ttl;
pub mod api_v2;

use engine_traits::Result;
use kvproto::kvrpcpb::ApiVersion;
pub use match_template::match_template;

pub trait KvFormat: Clone + Copy + 'static + Send + Sync {
    const TAG: ApiVersion;
    /// Corresponding TAG of client requests. For test only.
    #[cfg(any(test, feature = "testexport"))]
    const CLIENT_TAG: ApiVersion;
    const IS_TTL_ENABLED: bool;

    /// Parse the key prefix and infer key mode. It's safe to parse either raw
    /// key or encoded key.
    fn parse_key_mode(key: &[u8]) -> KeyMode;
    fn parse_range_mode(range: (Option<&[u8]>, Option<&[u8]>)) -> KeyMode;

    fn strip_keyspace(key: &[u8]) -> Result<(Option<u32>, &[u8])> {
        Ok((None, key))
    }
}

#[derive(Default, Clone, Copy)]
pub struct ApiV1;
#[derive(Default, Clone, Copy)]
pub struct ApiV1Ttl;
#[derive(Default, Clone, Copy)]
pub struct ApiV2;

#[macro_export]
macro_rules! test_kv_format_impl {
    ($func:ident<$ver:ident $($left_ver:ident)*> $(($($arg:expr),*))?) => {
        $crate::test_kv_format_impl!(__imp $func<$ver> $(($($arg),*))?);
        $crate::test_kv_format_impl!($func<$($left_ver)*> $(($($arg),*))?);
    };
    ($func:ident<> $(($($arg:expr),*))?) => {
    };
    ($func:ident $(($($arg:expr),*))?) => {
        $crate::test_kv_format_impl!($func<ApiV1 ApiV1Ttl ApiV2>$(($($arg),*))?);
    };
    (__imp $func:ident<$ver:ident>) => {
        $func::<$crate::$ver>();
    };
    (__imp $func:ident<$ver:ident>($($arg:expr),*)) => {
        $func::<$crate::$ver>($($arg),*);
    };
}

// TODO: move `match_template_api_version!` usage to `dispatch_api_version!`.
#[macro_export]
macro_rules! match_template_api_version {
     ($t:tt, $($tail:tt)*) => {{
         $crate::match_template! {
             $t = [
                V1 => $crate::ApiV1,
                V1ttl => $crate::ApiV1Ttl,
                V2 => $crate::ApiV2,
            ],
            $($tail)*
         }
     }}
}

/// Dispatch an expression with type `kvproto::kvrpcpb::ApiVersion` to
/// corresponding concrete type of `KvFormat`
///
/// For example, the following code
///
/// ```ignore
/// let encoded_key = dispatch_api_version(api_version, API::encode_raw_key(key));
/// ```
///
/// generates
///
/// ```ignore
/// let encoded_key = match api_version {
///     ApiVersion::V1 => ApiV1::encode_raw_key(key),
///     ApiVersion::V1ttl => ApiV1Ttl::encode_raw_key(key),
///     ApiVersion::V2 => ApiV2::encode_raw_key(key),
/// };
/// ```
#[macro_export]
macro_rules! dispatch_api_version {
    ($api_version:expr, $e:expr) => {
        $crate::match_template! {
            API = [
                V1 => $crate::ApiV1,
                V1ttl => $crate::ApiV1Ttl,
                V2 => $crate::ApiV2,
            ],
            match $api_version {
                kvproto::kvrpcpb::ApiVersion::API => $e,
            }
        }
    };
}

/// The key mode inferred from the key prefix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyMode {
    /// Raw key.
    Raw,
    /// Transaction key.
    Txn,
    /// TiDB key.
    ///
    /// It doesn't mean that the key is certainly written by
    /// TiDB, but instead, it means that the key matches the definition of
    /// TiDB key in API V2, therefore, the key is treated as TiDB data in
    /// order to fulfill compatibility.
    Tidb,
    /// Unrecognised key mode.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_v2::*;

    #[test]
    fn test_parse() {
        assert_eq!(ApiV1::parse_key_mode(&b"t_a"[..]), KeyMode::Unknown);
        assert_eq!(ApiV1Ttl::parse_key_mode(&b"ot"[..]), KeyMode::Raw);
        assert_eq!(
            ApiV2::parse_key_mode(&[RAW_KEY_PREFIX, b'a', b'b', b'c']),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_key_mode(&[TXN_KEY_PREFIX, b'a', b'b', b'c']),
            KeyMode::Txn
        );
        assert_eq!(ApiV2::parse_key_mode(&b"t_a\0\0"[..]), KeyMode::Tidb);
        assert_eq!(ApiV2::parse_key_mode(&b"m\0\0\0"[..]), KeyMode::Tidb);
        assert_eq!(ApiV2::parse_key_mode(&b"ot"[..]), KeyMode::Unknown);
    }

    #[test]
    fn test_parse_range() {
        assert_eq!(ApiV1::parse_range_mode((None, None)), KeyMode::Unknown);
        assert_eq!(
            ApiV1::parse_range_mode((Some(b"x"), None)),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV1Ttl::parse_range_mode((Some(b"m_a"), Some(b"na"))),
            KeyMode::Raw
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"t_a\0"), Some(b"t_z\0"))),
            KeyMode::Tidb
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"t\0\0\0"), Some(b"u"))),
            KeyMode::Tidb
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"m\0\0\0"), Some(b"n"))),
            KeyMode::Tidb
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"m_a\0"), Some(b"m_z\0"))),
            KeyMode::Tidb
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"x\0\0a"), Some(b"x\0\0z"))),
            KeyMode::Txn
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"x\0\0\0"), Some(b"y"))),
            KeyMode::Txn
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"r\0\0a"), Some(b"r\0\0z"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"r\0\0\0"), Some(b"s"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"t_a"), Some(b"ua"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"t"), None)),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((None, Some(b"t_z"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"m_a"), Some(b"na"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"m"), None)),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((None, Some(b"m_z"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"x\0a"), Some(b"ya"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"x"), None)),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((None, Some(b"x\0z"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"r\0a"), Some(b"sa"))),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((Some(b"r"), None)),
            KeyMode::Unknown
        );
        assert_eq!(
            ApiV2::parse_range_mode((None, Some(b"r\0z"))),
            KeyMode::Unknown
        );
    }
}
