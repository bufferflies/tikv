// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

#![cfg(all(debug_assertions, feature = "debug-trace-ia-segments"))]

use std::sync::Arc;

use crossbeam::queue::ArrayQueue;
use dashmap::DashMap;

use crate::{
    ia::types::{FileSegmentData, FileSegmentIdent},
    table::Result,
};

const SEGMENT_ACTION_HISTORY_CAP: usize = 1024;

#[derive(Debug, Clone)]
pub(crate) enum SegmentAction {
    InsertSmall,
    EvictFromSmall,
    InsertMain,
    EvictFromMain,
    InsertGhost,
    InsertSegmentDataToMemory,
    InsertSegmentDataToStore,
    RemoveSegmentDataFromMemory,
    RemoveSegmentDataFromStore,
    RemoveFromMainStore(Result<Option<()>>),
    MoveToMainStore(Result<()>),
    PostMoveToMainStore(std::result::Result<(), Option<FileSegmentData>>),
    SkipMoveAlreadyInStore,
    SkipMoveNotCached,
}

lazy_static::lazy_static! {
    static ref SEGMENTS_ACTION_HISTORY: DashMap<FileSegmentIdent, Arc<ArrayQueue<SegmentAction>>> = DashMap::default();
}

pub(crate) fn trace_segment_action(ident: FileSegmentIdent, action: SegmentAction) {
    let history = SEGMENTS_ACTION_HISTORY
        .entry(ident)
        .or_insert_with(|| Arc::new(ArrayQueue::new(SEGMENT_ACTION_HISTORY_CAP)))
        .clone();
    history.force_push(action);
}

pub(crate) fn trace_insert_segment_data(
    ident: FileSegmentIdent,
    segment_data: Option<&FileSegmentData>,
) {
    match segment_data {
        Some(FileSegmentData::InMem(_)) => {
            trace_segment_action(ident, SegmentAction::InsertSegmentDataToMemory)
        }
        Some(FileSegmentData::InStore) => {
            trace_segment_action(ident, SegmentAction::InsertSegmentDataToStore)
        }
        None => {}
    }
}

pub(crate) fn trace_remove_segment_data(
    ident: FileSegmentIdent,
    segment_data: Option<&FileSegmentData>,
) {
    match segment_data {
        Some(FileSegmentData::InMem(_)) => {
            trace_segment_action(ident, SegmentAction::RemoveSegmentDataFromMemory)
        }
        Some(FileSegmentData::InStore) => {
            trace_segment_action(ident, SegmentAction::RemoveSegmentDataFromStore)
        }
        None => {}
    }
}

pub(crate) fn dump_segment_action_history(ident: FileSegmentIdent) -> Vec<SegmentAction> {
    let Some(history) = SEGMENTS_ACTION_HISTORY.remove(&ident) else {
        return vec![];
    };
    let history = history.1;
    let mut history_vec = Vec::with_capacity(history.len());
    while let Some(action) = history.pop() {
        history_vec.push(action);
    }
    history_vec
}
