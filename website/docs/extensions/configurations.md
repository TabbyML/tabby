---
sidebar_position: 98
---

# Advanced Configuration

This document describes the available configurations for Tabby IDE extensions.

## Config File

The Tabby agent, which is the core component of Tabby IDE extensions, reads configurations from the `~/.tabby-client/agent/config.toml` file. This file is automatically created when you first run the Tabby IDE extensions. You can edit this file to modify the configurations. The Tabby IDE extensions will automatically reload the config file when it detects changes.

:::tip
- Configurations set via the IDE settings page take precedence over the config file. If you want to use a configuration from the config file, make sure that the IDE setting is empty.
- If you are using the Tabby VSCode extension in a web browser, this config file is not available. You can use the VSCode settings page to configure the extension.
:::

## Server

The `server` section contains configurations related to the Tabby server.

```toml
# Server
# You can set the server endpoint here and an optional authentication token if required.
[server]
endpoint = "http://localhost:8080" # http or https URL
token = "your-token-here" # if token is set, request header Authorization = "Bearer $token" will be added automatically

# You can add custom request headers.
[server.requestHeaders]
Header1 = "Value1" # list your custom headers here
Header2 = "Value2" # values can be strings, numbers or booleans
```

## Completion

If you have changed the default response timeout at Tabby server side, you may also need to change the timeout configurations here.

```toml
# Completion
# (Since 1.1.0) You can set the completion request timeout here.
# Note that there is also a timeout config at the server side.
[completion]
timeout = 4000 # 4s
```

## Logs

If you encounter any issues with the Tabby IDE extensions and need to report a bug, you can enable debug logs to help us investigate the issue.

```toml
# Logs
# You can set the log level here. The log file is located at ~/.tabby-client/agent/logs/.
[logs]
level = "silent" # "silent" or "error" or "debug"
```

## Usage Collection

Tabby IDE extensions collect aggregated anonymous usage data and sends it to the Tabby team to help improve our products.

**Do not worry, your code, generated completions, or any identifying information is never tracked or transmitted.**  

The data we collect, as of the latest update on November 6, 2023, contains following major parts:

- System info and extension version info
- Completions statistics
  - Completion count
  - Completion accepted count
  - Completion HTTP request latency

We sincerely appreciate your contribution in sending anonymous usage data. However, if you prefer not to participate, you can disable anonymous usage tracking here:

```toml
# Anonymous usage tracking
[anonymousUsageTracking]
disable = false # set to true to disable
```
