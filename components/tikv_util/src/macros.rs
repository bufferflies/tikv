// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

//! The macros crate contains all useful needed macros.

/// A shortcut to box an error.
#[macro_export]
macro_rules! box_err {
    ($e:expr) => ({
        use std::error::Error;
        let e: Box<dyn Error + Sync + Send> = format!("[{}:{}]: {}", file!(), line!(),  $e).into();
        e.into()
    });
    ($f:tt, $($arg:expr),+) => ({
        box_err!(format!($f, $($arg),+))
    });
}

/// Boxes error first, and then does the same thing as `try!`.
#[macro_export]
macro_rules! box_try {
    ($expr:expr) => {{
        match $expr {
            Ok(r) => r,
            Err(e) => return Err($crate::box_err!(e)),
        }
    }};
}

/// Logs slow operations by `warn!`.
/// The final log level depends on the given `cost` and `slow_log_threshold`
#[macro_export]
macro_rules! slow_log {
    (T $t:expr, $($arg:tt)*) => {{
        if $t.is_slow() {
            warn!(#"slow_log_by_timer", $($arg)*; "takes" => $crate::logger::LogCost($crate::time::duration_to_ms($t.saturating_elapsed())));
        }
    }};
    ($n:expr, $($arg:tt)*) => {{
        warn!(#"slow_log", $($arg)*; "takes" => $crate::logger::LogCost($crate::time::duration_to_ms($n)));
    }}

}

/// Makes a thread name with an additional tag inherited from the current
/// thread.
#[macro_export]
macro_rules! thd_name {
    ($name:expr) => {{
        $crate::get_tag_from_thread_name()
            .map(|tag| format!("{}::{}", $name, tag))
            .unwrap_or_else(|| $name.to_owned())
    }};
}

/// Simulates Go's defer.
///
/// Please note that, different from go, this defer is bound to scope.
/// When exiting the scope, its deferred calls are executed in last-in-first-out
/// order.
#[macro_export]
macro_rules! defer {
    ($t:expr) => {
        let __ctx = $crate::DeferContext::new(|| $t);
    };
}

/// Waits for async operation. It returns `Option<Res>` after the expression
/// gets executed. It only accepts a `Result` expression.
#[macro_export]
macro_rules! wait_op {
    ($expr:expr) => {
        wait_op!(IMPL $expr, None)
    };
    ($expr:expr, $timeout:expr) => {
        wait_op!(IMPL $expr, Some($timeout))
    };
    (IMPL $expr:expr, $timeout:expr) => {{
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel();
        let cb = Box::new(move |res| {
            // we don't care error actually.
            let _ = tx.send(res);
        });
        $expr(cb)?;
        match $timeout {
            None => rx.recv().ok(),
            Some(timeout) => rx.recv_timeout(timeout).ok(),
        }
    }};
}

/// Checks `Result<Option<T>>`, and returns early when it meets `Err` or
/// `Ok(None)`.
#[macro_export]
macro_rules! try_opt {
    ($expr:expr) => {{
        match $expr {
            Err(e) => return Err(e.into()),
            Ok(None) => return Ok(None),
            Ok(Some(res)) => res,
        }
    }};
}

/// Checks `Result<Option<T>>`, and returns early when it meets `Err` or
/// `Ok(None)`. return `Ok(or)` when met `Ok(None)`.
#[macro_export]
macro_rules! try_opt_or {
    ($expr:expr, $or:expr) => {{
        match $expr {
            Err(e) => return Err(e.into()),
            Ok(None) => return Ok($or),
            Ok(Some(res)) => res,
        }
    }};
}

/// A safe panic macro that prevents double panic.
///
/// You probably want to use this macro instead of `panic!` in a `drop` method.
/// It checks whether the current thread is unwinding because of panic. If it
/// is, log an error message instead of causing double panic.
#[macro_export]
macro_rules! safe_panic {
    () => ({
        safe_panic!("explicit panic")
    });
    ($msg:expr) => ({
        if std::thread::panicking() {
            error!(concat!($msg, ", double panic prevented"))
        } else {
            panic!($msg)
        }
    });
    ($fmt:expr, $($args:tt)+) => ({
        if std::thread::panicking() {
            error!(concat!($fmt, ", double panic prevented"), $($args)+)
        } else {
            panic!($fmt, $($args)+)
        }
    });
}

#[macro_export]
macro_rules! impl_format_delegate_newtype {
    ($t:ty) => {
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_display_as_debug {
    ($t:ty) => {
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self)
            }
        }
    };
}

/// Transaction debug logging with category-based control.
///
/// This macro logs transaction-related debug information with category
/// filtering based on trace_control_flags from the request context. Logs are
/// output to INFO level if:
/// - The category is enabled via trace_control_flags, AND
/// - Either immediate_log flag is set OR global txn_info_logging is enabled
///
/// Otherwise logs go to DEBUG level and may be filtered out.
///
/// This macro supports only slog-style structured logging.
///
/// slog style: txn_debug!(trace_event::types::Category::ReqResp, "message";
/// "key1" => value1, "key2" => ?value2)
#[macro_export]
macro_rules! txn_debug {
    // slog-style: trace_ctx, category, message; key-value pairs
    (trace_ctx: $trace_ctx:expr, $category:expr, $msg:expr; $($args:tt)*) => {
        {
            let __trace_ctx = $trace_ctx;
            let __control_flags = __trace_ctx.control_flags();
            if __trace_ctx.category_enabled($category) {
                if __trace_ctx.immediate_log_enabled() {
                    info!($msg; "trace_id" => &__trace_ctx.trace_id, $($args)*);
                } else {
                    debug!($msg; "trace_id" => &__trace_ctx.trace_id, $($args)*);
                }
                $crate::txn_log_helper!(record __trace_ctx, $category, $msg; $($args)*)
            }
        }
    };

    // slog-style: category, message; key-value pairs
    ($category:expr, $msg:expr; $($args:tt)*) => {
        txn_debug!(trace_ctx: tracker::get_tls_trace_ctx(), $category, $msg; $($args)*);
    };
}

/// Logs operational information at INFO level.
/// Controlled by log.txn-info-logging config or per-request immediate_log flag.
/// Only supports slog-style syntax: message; key-value pairs
#[macro_export]
macro_rules! txn_info {
    (trace_ctx: $trace_ctx:expr, $category:expr, $msg:expr; $($args:tt)*) => {
        {
            let __trace_ctx = $trace_ctx;
            let __control_flags = __trace_ctx.control_flags();
            if __trace_ctx.immediate_log_enabled() || $crate::logger::txn_info_logging_enabled() {
                info!($msg; "trace_id" => &__trace_ctx.trace_id, $($args)*);
            }
            if __trace_ctx.category_enabled($category) {
                $crate::txn_log_helper!(record __trace_ctx, $category, $msg; $($args)*)
            }
        }
    };
    ($category:expr, $msg:expr; $($args:tt)*) => {
        txn_info!(trace_ctx: tracker::get_tls_trace_ctx(), $category, $msg; $($args)*);
    };
}

#[macro_export]
macro_rules! txn_log_helper {
    (record $trace_ctx:expr, $category:expr, $msg:expr; $($args:tt)*) => {
        {
            let __r = trace_event::flight_recorder::get_global_flight_recorder();
            if let Some(__r) = __r.as_ref() {
                let mut __events = Vec::with_capacity($crate::txn_log_helper!(count_fields $($args)*,));
                $crate::txn_log_helper!(push_events __events; $($args)*,);
                __r.record(trace_event::types::Event {
                    category: $category,
                    name: $msg,
                    trace_ctx: $trace_ctx,
                    time: std::time::SystemTime::now(),
                    fields: __events,
                });
            }
        }
    };

    // Recursively calculate the number of fields.
    (count_fields $($ctl:ident)? $name:expr => $(?)? $(%)? $value:expr, $($rem:tt)*) => {
        1 + $crate::txn_log_helper!(count_fields $($rem)*)
    };
    (count_fields $(,)?) => { 0 };

    // Recursively make EventField out from the fields.
    (push_events $events:ident; $name:expr => $(%)? $value:expr, $($rem:tt)*) => {
        $events.push(trace_event::types::EventField::new($name, &($value)));
        $crate::txn_log_helper!(push_events $events; $($rem)*);
    };
    (push_events $events:ident; $name:expr => ?$value:expr, $($rem:tt)*) => {
        $events.push(trace_event::types::EventField::new_with_owned_string($name, format!("{:?}", $value)));
        $crate::txn_log_helper!(push_events $events; $($rem)*);
    };
    (push_events $events:ident; $(,)?) => {}
}

/// Macro to consume Arc runtimes and call shutdown_background() if the Arc
/// is uniquely owned. Usage:
/// shutdown_runtimes!(arc_rt1, arc_rt2, ...);
#[macro_export]
macro_rules! shutdown_runtimes {
    ($($rt:expr),+ $(,)?) => {
        $(
            if let Some(rt) = std::sync::Arc::into_inner($rt) {
                rt.shutdown_background();
            }
        )+
    };
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use trace_event::{
        flight_recorder::{
            get_global_flight_recorder, set_global_flight_recorder,
            DEFAULT_GLOBAL_FLIGHT_RECORDER_CAPACITY,
        },
        types::{Category, ControlFlags, TraceContext, TraceId},
    };

    #[test]
    fn test_box_error() {
        let file_name = file!();
        let line_number = line!();
        let e: Box<dyn Error + Send + Sync> = box_err!("{}", "hi");
        assert_eq!(
            format!("{}", e),
            format!("[{}:{}]: hi", file_name, line_number + 1)
        );
    }

    #[test]
    fn test_safe_panic() {
        struct S;
        impl Drop for S {
            fn drop(&mut self) {
                safe_panic!("safe panic on drop");
            }
        }

        let res = panic_hook::recover_safe(|| {
            let _s = S;
            panic!("first panic");
        });
        res.unwrap_err();
    }

    #[test]
    fn test_txn_debug_recording_events() {
        tracker::set_tls_trace_ctx(TraceContext::from_proto(
            b"id1",
            ControlFlags::ENABLE_CATEGORY_REQ_RESP.bits(),
        ));

        set_global_flight_recorder(10);
        txn_debug!(Category::ReqResp, "test event";
            "k1" => 1u64,
            "k2" => "v2",
            "k3" => ?vec![1, 2, 3]
        );

        let mut collected_event = None;
        {
            let r = get_global_flight_recorder();
            r.as_ref().unwrap().for_each_event_by_trace_id(
                &Some(TraceId::from(b"id1".as_slice())),
                |e| {
                    collected_event = Some(e.clone());
                },
            );
        }
        let collected_event = collected_event.unwrap();
        assert_eq!(
            collected_event
                .trace_ctx
                .trace_id
                .as_ref()
                .unwrap()
                .0
                .as_ref(),
            b"id1"
        );
        assert_eq!(
            collected_event.trace_ctx.control_flags,
            ControlFlags::ENABLE_CATEGORY_REQ_RESP
        );
        assert_eq!(collected_event.category, Category::ReqResp);
        assert_eq!(collected_event.name, "test event");

        assert_eq!(collected_event.fields.len(), 3);
        assert_eq!(collected_event.fields[0].name(), "k1");
        assert_eq!(collected_event.fields[0].value().to_string(), "1");
        assert_eq!(collected_event.fields[1].name(), "k2");
        assert_eq!(collected_event.fields[1].value().to_string(), "v2");
        assert_eq!(collected_event.fields[2].name(), "k3");
        assert_eq!(collected_event.fields[2].value().to_string(), "[1, 2, 3]");

        set_global_flight_recorder(DEFAULT_GLOBAL_FLIGHT_RECORDER_CAPACITY);
    }
}
