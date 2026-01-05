// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

//! This module contains utilities to manage and retrieve the health status of
//! TiKV instance in a unified way.
//!
//! ## [`HealthController`]
//!
//! [`HealthController`] is the core of the module. It's a unified place where
//! the server's health status is managed and collected, including the [gRPC
//! `HealthService`](grpcio_health::HealthService). It provides interfaces to
//! retrieve the collected information, and actively setting whether
//! the gRPC `HealthService` should report a `Serving` or `NotServing` status.
//!
//! ## Reporters
//!
//! [`HealthController`] doesn't provide ways to update most of the states
//! directly. Instead, each module in TiKV tha need to report its health status
//! need to create a corresponding reporter.
//!
//! The reason why the reporters is split out from the `HealthController` is:
//!
//! * Reporters can have different designs to fit the special use patterns of
//!   different modules.
//! * `HealthController` internally contains states that are shared in different
//!   modules and threads. If some module need to store internal states to
//!   calculate the health status, they can be put in the reporter instead of
//!   the `HealthController`, which makes it possible to avoid unnecessary
//!   synchronization like mutexes.
//! * To avoid the `HealthController` itself contains too many different APIs
//!   that are specific to different modules, increasing the complexity and
//!   possibility to misuse of `HealthController`.

#![feature(div_duration)]

pub mod metrics;
pub mod reporters;
pub mod slow_score;
pub mod space_usage;
pub mod types;

use std::{
    collections::HashSet,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use collections::HashMap;
use grpcio_health::HealthService;
use parking_lot::Mutex;
pub use types::{InspectDuration, InspectFactor, LatencyInspector};

/// Trait for health checker to abstract network latency monitoring
pub trait HealthChecker: Send + Sync {
    /// Get all maximum latencies
    /// Returns a HashMap of store_id -> max_latency_ms
    fn get_all_max_latencies(&self) -> HashMap<u64, f64>;
}

struct ServingStatus {
    is_serving: bool,
    unhealthy_modules: HashSet<&'static str>,
}

impl ServingStatus {
    fn to_serving_status_pb(&self) -> grpcio_health::ServingStatus {
        match (self.is_serving, self.unhealthy_modules.is_empty()) {
            (true, true) => grpcio_health::ServingStatus::Serving,
            (true, false) => grpcio_health::ServingStatus::ServiceUnknown,
            (false, _) => grpcio_health::ServingStatus::NotServing,
        }
    }
}

struct HealthControllerInner {
    // Internally stores a `f64` type.
    rfstore_slow_score: AtomicU64,

    /// Optional raft client health checker for network latency monitoring
    health_checker: Mutex<Option<Box<dyn HealthChecker>>>,

    /// gRPC's builtin `HealthService`.
    ///
    /// **Note**: DO NOT update its state directly. Only change its state while
    /// holding the mutex of `current_serving_status`, and keep consistent
    /// with value of `current_serving_status`, unless `health_service` is
    /// already shutdown.
    ///
    /// TiKV uses gRPC's builtin `HealthService` to provide information about
    /// whether the TiKV server is normally running. To keep its behavior
    /// consistent with earlier versions without the `HealthController`,
    /// it's used in such pattern:
    ///
    /// * Only an empty service name is used, representing the status of the
    ///   whole server.
    /// * When `current_serving_status.is_serving` is set to false (by calling
    ///   [`set_is_serving(false)`](HealthController::set_is_serving)), the
    ///   serving status is set to `NotServing`.
    /// * If `current_serving_status.is_serving` is true, but
    ///   `current_serving_status.unhealthy_modules` is not empty, the serving
    ///   status is set to `ServiceUnknown`.
    /// * Otherwise, the TiKV instance is regarded operational and the serving
    ///   status is set to `Serving`.
    health_service: HealthService,
    current_serving_status: Mutex<ServingStatus>,
}

impl HealthControllerInner {
    fn new() -> Self {
        let health_service = HealthService::default();
        health_service.set_serving_status("", grpcio_health::ServingStatus::NotServing);
        Self {
            rfstore_slow_score: AtomicU64::new(f64::to_bits(1.0)),
            health_checker: Mutex::new(None),

            health_service,
            current_serving_status: Mutex::new(ServingStatus {
                is_serving: false,
                unhealthy_modules: HashSet::default(),
            }),
        }
    }

    /// Marks a module (identified by name) to be unhealthy. Adding an unhealthy
    /// will make the serving status of the TiKV server, reported via the
    /// gRPC `HealthService`, to become `ServiceUnknown`.
    ///
    /// This is not an public API. This method is expected to be called only
    /// from reporters.
    fn add_unhealthy_module(&self, module_name: &'static str) {
        let mut status = self.current_serving_status.lock();
        if !status.unhealthy_modules.insert(module_name) {
            // Nothing changed.
            return;
        }
        if status.unhealthy_modules.len() == 1 && status.is_serving {
            debug_assert_eq!(
                status.to_serving_status_pb(),
                grpcio_health::ServingStatus::ServiceUnknown
            );
            self.health_service
                .set_serving_status("", grpcio_health::ServingStatus::ServiceUnknown);
        }
    }

    /// Removes a module (identified by name) that was marked unhealthy before.
    /// When the unhealthy modules are cleared, the serving status reported
    /// via the gRPC `HealthService` will change from `ServiceUnknown` to
    /// `Serving`.
    ///
    /// This is not an public API. This method is expected to be called only
    /// from reporters.
    fn remove_unhealthy_module(&self, module_name: &'static str) {
        let mut status = self.current_serving_status.lock();
        if !status.unhealthy_modules.remove(module_name) {
            // Nothing changed.
            return;
        }
        if status.unhealthy_modules.is_empty() && status.is_serving {
            debug_assert_eq!(
                status.to_serving_status_pb(),
                grpcio_health::ServingStatus::Serving
            );
            self.health_service
                .set_serving_status("", grpcio_health::ServingStatus::Serving);
        }
    }

    /// Sets whether the TiKV server is serving. This is currently used to pause
    /// the server, which has implementation in code but not commonly used.
    ///
    /// The effect of setting not serving overrides the effect of
    /// [`add_on_healthy_module`](Self::add_unhealthy_module).
    fn set_is_serving(&self, is_serving: bool) {
        let mut status = self.current_serving_status.lock();
        if is_serving == status.is_serving {
            // Nothing to do.
            return;
        }
        status.is_serving = is_serving;
        self.health_service
            .set_serving_status("", status.to_serving_status_pb());
    }

    /// Gets the current serving status that is being reported by
    /// `health_service`, if it's not shutdown.
    fn get_serving_status(&self) -> grpcio_health::ServingStatus {
        let status = self.current_serving_status.lock();
        status.to_serving_status_pb()
    }

    fn update_rfstore_slow_score(&self, value: f64) {
        self.rfstore_slow_score
            .store(value.to_bits(), Ordering::Release);
    }

    fn get_rfstore_slow_score(&self) -> f64 {
        f64::from_bits(self.rfstore_slow_score.load(Ordering::Acquire))
    }

    fn shutdown(&self) {
        self.health_service.shutdown();
    }

    /// Set the health checker for network latency monitoring
    fn set_health_checker(&self, checker: Box<dyn HealthChecker>) {
        let mut health_checker_guard = self.health_checker.lock();
        *health_checker_guard = Some(checker);
    }

    /// Get network latencies from the health checker
    fn get_network_latencies(&self) -> HashMap<u64, Duration> {
        if let Some(checker) = self.health_checker.lock().as_ref() {
            checker
                .get_all_max_latencies()
                .into_iter()
                .map(|(store_id, latency_ms)| (store_id, Duration::from_millis(latency_ms as u64)))
                .collect()
        } else {
            HashMap::default()
        }
    }
}

#[derive(Clone)]
pub struct HealthController {
    inner: Arc<HealthControllerInner>,
}

impl HealthController {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(HealthControllerInner::new()),
        }
    }

    /// Set a health checker that implements HealthChecker
    pub fn set_health_checker(&self, checker: Box<dyn HealthChecker>) {
        self.inner.set_health_checker(checker);
    }

    pub fn get_rfstore_slow_score(&self) -> f64 {
        self.inner.get_rfstore_slow_score()
    }

    /// Get the gRPC `HealthService`.
    ///
    /// Only use this when it's necessary to startup the gRPC server or for test
    /// purpose. Do not change the `HealthService`'s state manually.
    ///
    /// If it's necessary to update `HealthService`'s state, consider using
    /// [`set_is_serving`](Self::set_is_serving) or use a reporter to add an
    /// unhealthy module. An example:
    ///  [`RaftstoreReporter::set_is_healthy`](reporters::RaftstoreReporter::set_is_healthy).
    pub fn get_grpc_health_service(&self) -> HealthService {
        self.inner.health_service.clone()
    }

    pub fn get_serving_status(&self) -> grpcio_health::ServingStatus {
        self.inner.get_serving_status()
    }

    /// Set whether the TiKV server is serving. This controls the state reported
    /// by the gRPC `HealthService`.
    pub fn set_is_serving(&self, is_serving: bool) {
        self.inner.set_is_serving(is_serving);
    }

    pub fn shutdown(&self) {
        self.inner.shutdown();
    }
}

// Make clippy happy.
impl Default for HealthControllerInner {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for HealthController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_health_controller_update_service_status() {
        let h = HealthController::new();

        // Initial value of slow score
        assert_eq!(h.get_rfstore_slow_score(), 1.0);

        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::NotServing
        );

        h.set_is_serving(true);
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::Serving
        );

        h.inner.add_unhealthy_module("A");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::ServiceUnknown
        );
        h.inner.add_unhealthy_module("B");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::ServiceUnknown
        );

        h.inner.remove_unhealthy_module("A");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::ServiceUnknown
        );
        h.inner.remove_unhealthy_module("B");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::Serving
        );

        h.set_is_serving(false);
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::NotServing
        );
        h.inner.add_unhealthy_module("A");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::NotServing
        );

        h.set_is_serving(true);
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::ServiceUnknown
        );

        h.inner.remove_unhealthy_module("A");
        assert_eq!(
            h.get_serving_status(),
            grpcio_health::ServingStatus::Serving
        );
    }
}
