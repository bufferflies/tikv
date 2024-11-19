# Recovery Mode

Recovery mode is a feature to prevent the whole cluster run into crash loop and become unavailable.

## Overview

When recovery mode is enabled:

1. All requests to keyspaces not in the white list will be rejected
2. All compaction tasks for the keyspaces not in the white list are discarded.
3. Rejected keyspaces are tracked and can be queried.
4. persistent black list can be maintained across restarts.

## Usage

The feature can be controlled via the `cse-ctl` command line tool:


### Config file

```toml
[pd]
# pd configurations

[security]
# security configurations

[dfs]
# dfs configurations
```


### Enable recovery mode

There are two ways to enable recovery mode, the first is to enable it after tikv-server is started:

```
cse-ctl --config config.toml recovery enable
```

Another way is to enable it by tikv config file by adding the following line to the config file:

```toml
recover-mode = true
```

Then tikv-server will enter recovery mode after restarts.

### Status
After recovery mode is enabled, we can use the following command to check the status of recovery mode.

```
./cse-ctl recovery --config config.toml status
```

The execution result looks like:
```
store: 1
in recovery mode:true
rejected:[123,125,148]
white_list:[]
black_list:[]
```

### Add white list
We can use the rejected keyspace ids to add white list.
The keyspaces in the white list will be allowed to execute requests.

```
./cse-ctl recovery --config config.toml --keyspace-ids 123,125 add-white-list
```

### Add black list
If we add a keyspace to white list then the cluster crashed soon, it is likely that the keyspace be 
the cause of the crash. We can add the keyspace to the black list to prevent future crashes.

```
./cse-ctl recovery --config config.toml --keyspace-ids 123,125 add-black-list
```

The black list take effect after tikv server restarts.

### Remove black list
When we fix the crashing bug or clean up the data of the black list keyspace, we can remove it from the black list.

```
./cse-ctl recovery --config config.toml --keyspace-ids 123,125 remove-black-list
```
