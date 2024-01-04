# Docker-Random

**Docker-Random** is a tool that allows you to run random tests using **Docker** to create an isolated environment. This collection of scripts facilitates the execution of various random tests, helping us to discover potential issues and vulnerabilities.

## Requirements

To use Docker-Random, ensure that Docker is installed on your system. Docker provides the necessary containerization capabilities to create isolated environments for testing.

## Usage

1. **Get the Code**: Clone the code from the [cloud-storage-engine](https://github.com/tidbcloud/cloud-storage-engine) repository.

2. **Build the Executable**:

    ```shell
    cd scripts/docker-random
    ./make-bin.sh
    ```

    Note that the executable is built as release target by default. If you wish to build it as debug target, execute the following command:

    ```shell
    ./make-bin.sh --debug
    ```

3. **Run the Tests**: After building the executable, execute the random tests by running the main script:

    ```shell
    ./docker-run-random.sh
    ```

    This will initiate the test execution process, creating Docker containers and running the tests.

    You can also specify the *CPU*, *MEMORY*, and *CONCURRENCY* for the tests:

    ```shell
    export CONCURRENCY=12
    export CPU=4
    export MEMORY=5g
    ./docker-run-random.sh
    ```

    It is recommended to set *CPU* to a low value to generate more context switches and expose potential issues caused by incorrect coordination (or execution order) between tasks. For [`random_test_all`](https://github.com/tidbcloud/cloud-storage-engine/blob/cloud-engine/tests/random/test_all.rs), the recommended value is `4`, which is also the default value for *CPU*.

    Set *MEMORY* to an appropriate value as long as no OOM (Out of Memory) issues occur.

    Adjust *CONCURRENCY* based on the *CPU*, *MEMORY*, and available resources of your system.

    Other arguments:

    - `--tmp-path`: This specifies the path to the temporal storage path.

        By default, the data generated during testing is stored in the file system of the Docker runtime, which is typically located at `/var/lib/docker/overlay2`. However, if you prefer to run the tests on a different disk, such as a faster NVMe disk, you can specify the `--tmp-path` argument:

        ```shell
        ./docker-run-random.sh --tmp-path /path/to/tmp
        ```

        Please note that it is recommended to avoid writing data and logs (which are located at `scripts/docker-random/logs`) to the same disk, as this could potentially cause performance issues.

    - `--test`: This allows you to specify the test case to be run. Currently available test cases are `all` and `with_tidb`.

        ```shell
        ./docker-run-random.sh --test all
        ```

    - `--tidb-version`: Set the version of TiDB (and PD) for `with_tidb` test case.

        ```shell
        ./docker-run-random.sh --test with_tidb --tidb-version v7.1.0
        ```

        Versions list can be found in [hub-new.pingcap.net/keyspace/tidb](https://hub-new.pingcap.net/harbor/projects/116/repositories/tidb).

    - `--no-rebuild-image`: Do NOT rebuild the testing Docker image.

        By default we always rebuild the testing Docker image to ensure that we are testing on latest TiDB/PD. However, if you prefer not to rebuild the image, you can specify the `--no-rebuild-image` argument.

4. **Analyze the Results**: The logs of failed tests can be found in `scripts/docker-random/error-logs`. You can examine these logs to identify the causes of failure.

5. **Stop the Tests**: The tests will terminate after running for 10000 x *CONCURRENCY* iterations. If you wish to stop the tests before completion, execute the following command:

    ```shell
    export CONCURRENCY=12
    ./docker-stop-random.sh
    ```

    Note that the environment variable of *CONCURRENCY* and `--test` argument must be the same with the values when running `docker-run-random.sh`.
