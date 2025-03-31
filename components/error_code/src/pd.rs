// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

define_error_codes!(
    "KV:Pd:",

    IO => ("Io", "", ""),
    CLUSTER_BOOTSTRAPPED => ("ClusterBootstraped", "", ""),
    CLUSTER_NOT_BOOTSTRAPPED => ("ClusterNotBootstraped", "", ""),
    INCOMPATIBLE => ("Imcompatible", "", ""),
    GRPC => ("Grpc", "", ""),
    STREAM_DISCONNECT => ("StreamDisconnect", "", ""),
    REGION_NOT_FOUND => ("RegionNotFound", "", ""),
    STORE_TOMBSTONE => ("StoreTombstone", "", ""),
    GLOBAL_CONFIG_NOT_FOUND => ("GlobalConfigNotFound", "", ""),
    DATA_COMPACTED => ("DataCompacted","",""),
    STALE_SERVICE_GC_SAFE_POINT => ("StaleServiceGcSafePoint", "", ""),
    CHANNEL_DROP => ("ChannelDrop", "", ""),
    TSO_SERVER_NOT_FOUND => ("TsoServerNotFound", "", ""),
    UNKNOWN => ("Unknown", "", "")
);
