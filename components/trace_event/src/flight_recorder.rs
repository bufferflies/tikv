// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{collections::VecDeque, ops::Deref, sync::Arc};

use arc_swap::ArcSwapOption;
use lazy_static::lazy_static;
use parking_lot::Mutex;

use crate::types::*;

/// A very simple ring buffer implemented with mutex and `VecDeque`. Its purpose
/// is to quickly provide the necessary functionality, and we will need
/// optimization soon.
struct RingBuffer<T> {
    inner: Mutex<VecDeque<T>>,
    max_capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(max_capacity: usize) -> Self {
        assert_ne!(max_capacity, 0);
        Self {
            inner: Mutex::new(VecDeque::with_capacity(max_capacity)),
            max_capacity,
        }
    }

    pub fn push(&self, item: T) {
        let mut buffer = self.inner.lock();
        if buffer.len() >= self.max_capacity {
            buffer.pop_front();
        }
        buffer.push_back(item);
    }

    pub fn for_each(&self, f: impl FnMut(&T)) {
        let buffer = self.inner.lock();
        buffer.iter().for_each(f);
    }
}

pub struct FlightRecorder {
    buffer: RingBuffer<Event>,
}

impl FlightRecorder {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: RingBuffer::new(capacity),
        }
    }

    pub fn record(&self, event: Event) {
        if event.needs_recording() {
            self.buffer.push(event);
        }
    }

    pub fn for_each_event(&self, f: impl FnMut(&Event)) {
        self.buffer.for_each(f);
    }

    pub fn for_each_event_by_trace_id(
        &self,
        trace_id: &Option<TraceId>,
        mut f: impl FnMut(&Event),
    ) {
        self.buffer.for_each(|e| {
            if e.trace_ctx.trace_id == *trace_id {
                f(e);
            }
        });
    }
}

lazy_static! {
    static ref GLOBAL_FLIGHT_RECORDER: ArcSwapOption<FlightRecorder> = ArcSwapOption::from(None);
}

// TODO: Set to non-zero value when its ready for production.
pub const DEFAULT_GLOBAL_FLIGHT_RECORDER_CAPACITY: usize = 0;

pub fn set_global_flight_recorder(capacity: usize) {
    // TODO: Maybe needs to keep the already-collected content when dynamically
    // changing the capacity.
    let new_recorder = if capacity > 0 {
        Some(Arc::new(FlightRecorder::new(capacity)))
    } else {
        None
    };
    GLOBAL_FLIGHT_RECORDER.store(new_recorder);
}

pub fn get_global_flight_recorder() -> impl Deref<Target = Option<Arc<FlightRecorder>>> {
    GLOBAL_FLIGHT_RECORDER.load()
}

#[cfg(test)]
mod tests {
    use std::time::SystemTime;

    use byteorder::{BigEndian, ByteOrder};

    use super::*;

    #[test]
    fn test_global_flight_recorder() {
        set_global_flight_recorder(0);
        assert!(get_global_flight_recorder().is_none());
        set_global_flight_recorder(10);
        {
            let r = get_global_flight_recorder();
            assert_eq!(r.as_ref().unwrap().buffer.max_capacity, 10);
        }
        set_global_flight_recorder(0);
        assert!(get_global_flight_recorder().is_none());
        set_global_flight_recorder(DEFAULT_GLOBAL_FLIGHT_RECORDER_CAPACITY);
        {
            let r = get_global_flight_recorder();
            // TODO: When the default capacity is changed to non-zero, update the test
            // assert_eq!(
            //     r.as_ref().unwrap().buffer.max_capacity,
            //     DEFAULT_GLOBAL_FLIGHT_RECORDER_CAPACITY
            // );
            assert!(r.is_none());
        }
    }

    #[test]
    fn test_ring_buffer() {
        let b = RingBuffer::new(5);
        assert_eq!(b.max_capacity, 5);
        assert!(b.inner.lock().capacity() >= 5);

        let collect_by_for_each = || {
            let mut res = vec![];
            b.for_each(|item| res.push(*item));
            res
        };

        assert!(collect_by_for_each().is_empty());

        for i in 0..5 {
            b.push(i);
        }
        {
            let buffer = b.inner.lock();
            assert_eq!(
                buffer.iter().cloned().collect::<Vec<_>>(),
                vec![0, 1, 2, 3, 4]
            );
        }
        assert_eq!(collect_by_for_each(), vec![0, 1, 2, 3, 4]);

        b.push(5);
        {
            let buffer = b.inner.lock();
            assert_eq!(
                buffer.iter().cloned().collect::<Vec<_>>(),
                vec![1, 2, 3, 4, 5]
            );
        }
        assert_eq!(collect_by_for_each(), vec![1, 2, 3, 4, 5]);

        for i in 6..10 {
            b.push(i);
        }
        {
            let buffer = b.inner.lock();
            assert_eq!(
                buffer.iter().cloned().collect::<Vec<_>>(),
                vec![5, 6, 7, 8, 9]
            );
        }
        assert_eq!(collect_by_for_each(), vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_flight_recorder() {
        let r = FlightRecorder::new(5);
        // The actual format is still TBD. This only satisfies the test temporarily.
        let make_ctx = |version: u8, ts: u64, rnd: u64| {
            let mut buffer = vec![];
            buffer.push(version);
            buffer.resize(buffer.len() + 16, 0);
            BigEndian::write_u64(&mut buffer[1..9], ts);
            BigEndian::write_u64(&mut buffer[9..17], rnd);
            TraceContext::from_proto(&buffer, ControlFlags::default().bits())
        };

        let make_event = |trace_ctx, category, name| Event {
            category,
            name,
            trace_ctx,
            time: SystemTime::now(),
            fields: vec![],
        };

        let extract_rnd = |trace_ctx: TraceContext| {
            BigEndian::read_u64(&trace_ctx.trace_id.as_ref().unwrap().0[9..17])
        };

        let collect_rnds = || {
            let mut res = vec![];
            r.for_each_event(|e| res.push(extract_rnd(e.trace_ctx.clone())));
            res
        };

        let collect_names = || {
            let mut res = vec![];
            r.for_each_event(|e| res.push(e.name));
            res
        };

        let count_by_trace_id = |trace_ctx: &TraceContext| {
            let mut count = 0;
            r.for_each_event_by_trace_id(&trace_ctx.trace_id, |_e| count += 1);
            count
        };

        assert!(collect_rnds().is_empty());
        assert_eq!(count_by_trace_id(&make_ctx(1, 100, 1000)), 0);

        r.record(make_event(
            make_ctx(1, 100, 1000),
            Category::ReqResp,
            "event1",
        ));
        assert_eq!(collect_rnds(), vec![1000]);
        assert_eq!(collect_names(), vec!["event1"]);

        r.record(make_event(
            make_ctx(1, 200, 1001),
            Category::ReqResp,
            "event2",
        ));
        assert_eq!(collect_rnds(), vec![1000, 1001]);
        assert_eq!(collect_names(), vec!["event1", "event2"]);

        r.record(make_event(
            make_ctx(1, 100, 1000),
            Category::ReqResp,
            "event3",
        ));
        assert_eq!(collect_rnds(), vec![1000, 1001, 1000]);
        assert_eq!(collect_names(), vec!["event1", "event2", "event3"]);

        assert_eq!(count_by_trace_id(&make_ctx(1, 100, 1000)), 2);
        assert_eq!(count_by_trace_id(&make_ctx(1, 200, 1001)), 1);
        assert_eq!(count_by_trace_id(&make_ctx(1, 100, 1002)), 0);

        for i in 0..4 {
            r.record(make_event(
                make_ctx(1, 300, 1002 + i as u64),
                Category::ReqResp,
                "event4",
            ));
        }

        assert_eq!(collect_rnds(), vec![1000, 1002, 1003, 1004, 1005]);
        assert_eq!(
            collect_names(),
            vec!["event3", "event4", "event4", "event4", "event4"]
        );
    }
}
