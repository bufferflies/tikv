# Docker-Random

**Docker-Random** is a tool that allows you to run random tests using **Docker** to create an isolated environment. This collection of scripts facilitates the execution of various random tests, helping us to discover potential issues and vulnerabilities.

## Requirements

To use Docker-Random, ensure that Docker is installed on your system. Docker provides the necessary containerization capabilities to create isolated environments for testing.

## Usage

1. **Get the Code**: Clone the code from the [cloud-storage-engine](https://github.com/tidbcloud/cloud-storage-engine) repository.

2. **Build the Executable**:

    ```
    cd scripts/docker-random
    ./make-bin.sh
    ```

3. **Run the Tests**: After building the executable, execute the random tests by running the main script:

    ```
    ./docker-run-random.sh
    ```

    This will initiate the test execution process, creating Docker containers and running the tests.

    You can also specify the *CPU*, *MEMORY*, and *CONCURRENCY* for the tests:

    ```
    export CONCURRENCY=12
    export CPU=4
    export MEMORY=5g
    ./docker-run-random.sh
    ```

    It is recommended to set *CPU* to a low value to generate more context switches and expose potential issues caused by incorrect coordination (or execution order) between tasks. For [`random_test_all`](https://github.com/tidbcloud/cloud-storage-engine/blob/cloud-engine/tests/random/test_all.rs), the recommended value is `4`, which is also the default value for *CPU*.

    Set *MEMORY* to an appropriate value as long as no OOM (Out of Memory) issues occur.

    Adjust *CONCURRENCY* based on the *CPU*, *MEMORY*, and available resources of your system.

4. **Analyze the Results**: The logs of failed tests can be found in `scripts/docker-random/error-logs`. You can examine these logs to identify the causes of failure.

5. **Stop the Tests**: The tests will terminate after running for 10000 x *CONCURRENCY* iterations. If you wish to stop the tests before completion, execute the following command:

    ```
    export CONCURRENCY=12
    ./docker-stop-random.sh
    ```

    Note that the value of *CONCURRENCY* must match the one used when running `docker-run-random.sh`.