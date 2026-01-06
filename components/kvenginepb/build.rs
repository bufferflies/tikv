// Copyright 2021 TiKV Project Authors. Licensed under Apache-2.0.

fn main() {
    println!("cargo:rerun-if-changed=src/changeset.proto");
    println!("cargo:rerun-if-changed=src/fts.proto");

    protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
        out_dir: "src",
        input: &["src/changeset.proto", "src/fts.proto"],
        includes: &["src"],
        customize: protobuf_codegen_pure::Customize {
            ..Default::default()
        },
    })
    .expect("protoc");

    // For some reason we must regenerate fts.proto separately, or there will be
    // some unstable name resolution issue.
    protobuf_codegen_pure::run(protobuf_codegen_pure::Args {
        out_dir: "src",
        input: &["src/fts.proto"],
        includes: &["src"],
        customize: protobuf_codegen_pure::Customize {
            ..Default::default()
        },
    })
    .expect("protoc");
}
