// Copyright 2025 TiKV Project Authors. Licensed under Apache-2.0.

use std::{sync::Arc, time::SystemTime};

use bitflags::bitflags;
use slog::{Record, Serializer, Value};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraceId(pub Arc<[u8]>);

impl From<&[u8]> for TraceId {
    fn from(bytes: &[u8]) -> Self {
        TraceId(Arc::from(bytes))
    }
}

#[derive(Clone, Default, Debug)]
pub struct TraceContext {
    pub trace_id: Option<TraceId>,
    pub control_flags: ControlFlags,
}

impl TraceContext {
    /// Creates a `TraceContext` from fields from protobuf.
    /// An empty `trace_id` is considered as not set. In this case,
    /// `control_flags` will also be considered not set, and will be set to
    /// default regardless of the given value.
    pub fn from_proto(trace_id: &[u8], control_flags: u64) -> TraceContext {
        if !trace_id.is_empty() {
            TraceContext {
                trace_id: Some(trace_id.into()),
                control_flags: ControlFlags::from_bits_truncate(control_flags),
            }
        } else {
            TraceContext {
                trace_id: None,
                // Assuming when trace_id is not set, the client won't set control_flags either.
                control_flags: ControlFlags::default(),
            }
        }
    }

    pub fn control_flags(&self) -> ControlFlags {
        self.control_flags
    }

    /// Checks if the trace context contains the flag indicating that it needs
    /// to be output immediately.
    pub fn immediate_log_enabled(&self) -> bool {
        self.control_flags.contains(ControlFlags::IMMEDIATE_LOG)
    }

    /// Checks if the trace context contains the flag that indicates the event
    /// needs to be recorded.
    pub fn category_enabled(&self, category: Category) -> bool {
        self.control_flags.contains(category.flag_bit())
    }
}

impl Value for TraceId {
    fn serialize(
        &self,
        _record: &Record<'_>,
        key: slog::Key,
        serializer: &mut dyn Serializer,
    ) -> slog::Result {
        serializer.emit_str(key, &hex::encode(&self.0))
    }
}

/// Represents the category of an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    ReqResp,
    WriteDetails,
    ReadDetails,
}

impl Category {
    /// Get the control flag for enabling the category.
    pub fn flag_bit(self) -> ControlFlags {
        match self {
            Category::ReqResp => ControlFlags::ENABLE_CATEGORY_REQ_RESP,
            Category::WriteDetails => ControlFlags::ENABLE_CATEGORY_WRITE_DETAILS,
            Category::ReadDetails => ControlFlags::ENABLE_CATEGORY_READ_DETAILS,
        }
    }
}

// Represents a value of an event field. Avoids unnecessarily evaluating strings
// for some simple types when it doesn't need to be logged.
#[derive(Debug, Clone)]
pub enum EventFieldValue {
    U64(u64),
    U32(u32),
    USize(usize),
    Bool(bool),
    String(String),
    None,
}

pub trait AsEventFieldValue {
    fn as_event_field_value(self) -> EventFieldValue;
}

impl AsEventFieldValue for u64 {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::U64(self)
    }
}

impl AsEventFieldValue for u32 {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::U32(self)
    }
}

impl AsEventFieldValue for usize {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::USize(self)
    }
}

impl AsEventFieldValue for bool {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::Bool(self)
    }
}

impl AsEventFieldValue for &str {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::String(self.to_string())
    }
}

impl AsEventFieldValue for &String {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::String(self.clone())
    }
}

impl AsEventFieldValue for log_wrappers::Value<'_> {
    fn as_event_field_value(self) -> EventFieldValue {
        EventFieldValue::String(format!("{}", self))
    }
}

impl<T: AsEventFieldValue> AsEventFieldValue for Option<T> {
    fn as_event_field_value(self) -> EventFieldValue {
        match self {
            Some(v) => v.as_event_field_value(),
            None => EventFieldValue::None,
        }
    }
}

// Handle nested references.
impl<T: AsEventFieldValue + Copy> AsEventFieldValue for &T {
    fn as_event_field_value(self) -> EventFieldValue {
        (*self).as_event_field_value()
    }
}

impl std::fmt::Display for EventFieldValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventFieldValue::U64(v) => write!(f, "{}", v),
            EventFieldValue::U32(v) => write!(f, "{}", v),
            EventFieldValue::USize(v) => write!(f, "{}", v),
            EventFieldValue::Bool(v) => write!(f, "{}", v),
            EventFieldValue::String(v) => write!(f, "{}", v),
            EventFieldValue::None => write!(f, "None"),
        }
    }
}

/// Represents a log field in an event. Lazy-evaluation is allowed.
#[derive(Clone)]
pub struct EventField {
    name: &'static str,
    value: EventFieldValue,
}

impl EventField {
    /// Create an `EventField` with the given name and value.
    pub fn new(name: &'static str, value: impl AsEventFieldValue) -> Self {
        EventField {
            name,
            value: value.as_event_field_value(),
        }
    }

    pub fn new_with_owned_string(name: &'static str, value: String) -> Self {
        EventField {
            name,
            value: EventFieldValue::String(value),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn value(&self) -> &EventFieldValue {
        &self.value
    }
}

impl std::fmt::Debug for EventField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventField")
            .field("name", &self.name)
            .field("value", &self.value)
            .finish()
    }
}

impl std::fmt::Display for EventField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

bitflags! {
    pub struct ControlFlags: u64 {
        const IMMEDIATE_LOG = 1;
        const ENABLE_CATEGORY_REQ_RESP = 1 << 1;
        const ENABLE_CATEGORY_WRITE_DETAILS = 1 << 2;
        const ENABLE_CATEGORY_READ_DETAILS = 1 << 3;
    }
}

impl Default for ControlFlags {
    fn default() -> Self {
        ControlFlags::ENABLE_CATEGORY_REQ_RESP
    }
}

#[derive(Debug, Clone)]
pub struct Event {
    pub trace_ctx: TraceContext,
    pub category: Category,
    pub name: &'static str,
    pub time: SystemTime,
    pub fields: Vec<EventField>,
}

impl Event {
    pub fn new(
        trace_ctx: TraceContext,
        category: Category,
        name: &'static str,
        time: SystemTime,
        fields: Vec<EventField>,
    ) -> Self {
        Event {
            trace_ctx,
            category,
            name,
            time,
            fields,
        }
    }

    /// Checks if the event contains the flag indicating that it needs to be
    /// output immediately.
    pub fn needs_immediately_log(&self) -> bool {
        self.trace_ctx.immediate_log_enabled()
    }

    /// Checks if the event contains the flag that indicates the event needs to
    /// be recorded.
    pub fn needs_recording(&self) -> bool {
        self.trace_ctx.category_enabled(self.category)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_ctx_control_flags() {
        // Test default behavior: empty trace context enables only ReqResp
        let t = TraceContext::from_proto(b"", 0);
        assert!(!t.immediate_log_enabled());
        assert!(t.category_enabled(Category::ReqResp));
        assert!(!t.category_enabled(Category::WriteDetails));
        assert!(!t.category_enabled(Category::ReadDetails));

        // Non-empty context with flag=0 disables all category.
        let t = TraceContext::from_proto(b"a", 0);
        assert!(!t.immediate_log_enabled());
        assert!(!t.category_enabled(Category::ReqResp));
        assert!(!t.category_enabled(Category::WriteDetails));
        assert!(!t.category_enabled(Category::ReadDetails));

        // Test immediate_log flag.
        let t = TraceContext::from_proto(b"a", ControlFlags::IMMEDIATE_LOG.bits());
        assert!(t.immediate_log_enabled());
        let t = TraceContext::from_proto(
            b"a",
            (ControlFlags::IMMEDIATE_LOG | ControlFlags::ENABLE_CATEGORY_WRITE_DETAILS).bits(),
        );
        assert!(t.immediate_log_enabled());

        // Test category flags.
        let t = TraceContext::from_proto(
            b"a",
            (ControlFlags::ENABLE_CATEGORY_WRITE_DETAILS
                | ControlFlags::ENABLE_CATEGORY_READ_DETAILS)
                .bits(),
        );
        assert!(!t.category_enabled(Category::ReqResp));
        assert!(t.category_enabled(Category::WriteDetails));
        assert!(t.category_enabled(Category::ReadDetails));

        // Test TraceCategory mapping.
        assert_eq!(
            Category::ReqResp.flag_bit(),
            ControlFlags::ENABLE_CATEGORY_REQ_RESP
        );
        assert_eq!(
            Category::WriteDetails.flag_bit(),
            ControlFlags::ENABLE_CATEGORY_WRITE_DETAILS
        );
        assert_eq!(
            Category::ReadDetails.flag_bit(),
            ControlFlags::ENABLE_CATEGORY_READ_DETAILS
        );
    }

    #[test]
    fn test_flags_in_event() {
        let create_mock_event = |category: Category, flags: u64| Event {
            trace_ctx: TraceContext::from_proto(b"a", flags),
            category,
            name: "event1",
            time: SystemTime::now(),
            fields: vec![],
        };

        let e = create_mock_event(Category::ReqResp, 0);
        assert!(!e.needs_immediately_log());
        assert!(!e.needs_recording());
        let e = create_mock_event(Category::ReqResp, 1);
        assert!(e.needs_immediately_log());
        assert!(!e.needs_recording());
        let e = create_mock_event(Category::ReqResp, 3);
        assert!(e.needs_immediately_log());
        assert!(e.needs_recording());
        let e = create_mock_event(Category::ReqResp, 4);
        assert!(!e.needs_immediately_log());
        assert!(!e.needs_recording());
        let e = create_mock_event(Category::WriteDetails, 4);
        assert!(!e.needs_immediately_log());
        assert!(e.needs_recording());
        let e = create_mock_event(Category::ReadDetails, 8);
        assert!(!e.needs_immediately_log());
        assert!(e.needs_recording());
        let e = create_mock_event(Category::ReadDetails, 0xf);
        assert!(e.needs_immediately_log());
        assert!(e.needs_recording());
        let e = create_mock_event(Category::WriteDetails, 0xa);
        assert!(!e.needs_immediately_log());
        assert!(!e.needs_recording());
    }
}
