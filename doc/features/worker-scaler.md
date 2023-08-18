# Worker Scaler

The Worker Scaler feature optimizes the performance of the tikv-worker deployment within a Kubernetes (K8s) cluster by dynamically adjusting the number of replicas based on the current workload conditions. 
This intelligent scaling ensures efficient resource utilization and responsive handling of varying loads.

## Configuration

Default configuration values are set as follows:

```toml
[worker-scaler]
run = false
namespace = "tidb-serverless"
label = "app.kubernetes.io/name=tikv-worker"
worker-port = 19000
cpu-max-ratio = 0.6
cpu-min-ratio = 0.4
sample-interval = 10.0
max-replicas = 64
min-replicas = 1
```

- The `run` option enables the Worker Scaler within a `tikv-worker` process.
- The Worker Scaler should be enabled on a single instance, typically on the `tikv-api` component.
- The `label` option helps identify and select the relevant `tikv-worker` pods in the K8s cluster.
- The `cpu-max-ratio` and `cpu-min-ratio` parameters influence the scaling algorithm, specifying the allowable CPU utilization range (0.0 to 1.0).
Scaling out occurs when the CPU utilization ratio surpasses `cpu-max-ratio`, while scaling in occurs when it falls below `cpu-min-ratio`.
- The `sample-interval` option, measured in seconds, determines the frequency at which metrics from `tikv-worker` pods are collected.
- The `max-replicas` parameter sets the upper limit for the number of replicas.
- The `min-replicas` parameter sets the lower limit for the number of replicas.

## Implementation

To initiate the Worker Scaler, use the `new_worker_scaler` function to create an instance, which can then be set to run in the background.

During each sample interval, the worker scaler performs the following steps:

- Retrieves a list of tikv-worker pods within the K8s cluster.
- Collects CPU usage metrics from each of these pods.
- Do scale-out or scale-in if needed.

Scaling actions are only triggered when an adequate number of CPU usage samples is available. 
Scaling out requires fewer samples due to the need to address sudden workload spikes. The scaling increment is proportionate to ensure quick adjustment to the desired replica count.
Scaling in, in contrast, demands more samples to prevent rapid fluctuations in scaling. Scaling in occurs gradually, reducing one replica at a time for stability.
By intelligently adjusting replica counts based on real-time workload conditions, the Worker Scaler enhances the efficiency and responsiveness of the tikv-worker deployment, contributing to a seamless and optimized system performance.