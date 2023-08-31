# Worker Scaler

The Worker Scaler feature dynamically spawn load_data_worker pods on receive load_data task, and clean up the 
load_data_worker pods when the load_data task is finished.

## Configuration

Default configuration values are set as follows:

```toml
[worker-scaler]
run = false
namespace = "tidb-serverless"
template-sts-name = "tikv-api"
name = "load-data-worker"
worker-port = 19500
max-size = "1TB"
spawn-data-size = "2GB"
spawn-running-tasks = 4
worker-min-storage-gb = 128
worker-max-cores = 8
expire-seconds = 600
worker-count-limit = 1024
```

- The `run` option runs the Worker-Scaler within a `tikv-worker` process.
- The `template-sts-name` option specifies the name of the stateful-set used as a template for spawning `load-data-worker` pods.
- The `name` option is used to name the spawned worker stateful-set.
- The `max-size` option limits the maximum data size to load, if the data size is larger than this value, we return error to the client.
- The `spawn-data-size` and `spawn-running-tasks` is used to control whether a load-data task execute locally on `tikv-api` pod or spawn a new `load-data-worker` to handle the task.
  if the task data-size is larger than `spawn-data-size-mb` or locally running tasks is greater than `spawn-running-tasks`, we spawn a new `load-data-worker`, otherwise the task is executed locally on the `tikv-api` pod.
- The `worker-max-cores` specifies the maximum cores for the `load-data-worker` pod.
- The `expire-seconds` specifies the expiration time for the `load-data-worker` pod, if the pod is not used for `expire-seconds`, we will delete the pod.
- The `worker-count-limit` limits the max running `load-data-worker` pods.

## Implementation

When the `tikv-api` process received a `load_data` initial request, we check if the task should be run locally or spawn a new `load-data-worker` 
pod to handle the task.
This determination is based on two factors: the `data_size` parameter provided in the request and the current count of tasks being run locally.

To spawn a new `load-data-worker` pod, we create a new stateful-set based on the `template-sts-name` option, and change the
new stateful-set name based on the unique task id, so each task creates a unique stateful-set.
The pod's cpu, memory and storage is set based on the `data_size` of the task.

When the load-data-worker pod is running, we return the pod's address to the client and the client will send the load_data
request to the new pod.

The worker-scaler periodically checks the `load-data-worker` pods, if the pod is not used for `expire-seconds`, we will
delete the stateful-set and the pvc.
