## 1.5.2

### Fixes

- Fixed an issue where the indexing worker for recently edited code may cause a stuck.

## 1.5.0

### Features

- Added support for sending additional context for completion requests, including:
  - Filepath
  - Git repository information
  - Relevant declaration code snippets
  - Relevant recently edited code snippets
- Merged output channels `Tabby` and `Tabby Agent` into one output channel `Tabby`.

### Fixes

- Corrected server-side config retrieval behavior for connections to Tabby servers with version < 0.9.

## 1.4.0

### Features

- Added support for loading system-wide CA certificates. Previously, only Node.js bundled CA certificates were used.
- Added support for loading configurations from Tabby server, including `Disabling Client-side Telemetry`.
- Added output channel for logging. The log level can be configured by VSCode command `Developer: Set Log Level...`.
- Removed the notification when disconnected from Tabby server, keep only status bar icon.

### Fixes

- Fixed keybinding for accepting inline completion next word for macOS (cmd+right).

## 1.3.2

### Fixes:

- Disabled experimental features by default:
  - Stripping auto-closing characters in prompt suffix.
  - Syntax-based code completion scope limit.

## 1.3.1

### Fixes:

- Disabled experimental feature of syntax-based replace range calculation by default.

## 1.3.0

### Features:

- Removed the completion request timeout limit. Now, a warning status bar icon will be displayed when the completion requests take too long.
- Enabled experimental feature of stripping auto-closing characters in prompt suffix by default.
- Enabled experimental feature of syntax-based post-processing by default.
- Added support for logging completion dismiss events.

### Fixes:

- Fixed an issue where completion was triggered when the text selection was not empty.
- Fixed a bug that caused the completion not to show in non-last cells of a notebook.
- Fixed health checking to be compatibility with Tabby server version 0.2.0 or earlier.

## 1.2.0

### Features:

- Added support for setting Tabby server token in VSCode.
  - You can still configure the token in the agent config file, but the token set in VSCode will take precedence.
- A notification will now be displayed when the server requires a token.
- Removed support for automatically opening the authentication page and fetching the token when using Tabby Cloud.
  - For connecting to Tabby Cloud server, you need to manually set the token instead. The token already in use will still be usable.

### Fixes:

- Corrected invalid online documentation links.
- Fixed a bug that the document context was not fully provided for completion when editing a jupyter notebook file.
- Resolved a bug that resulted in empty log files being generated even when the logging level is set to `silent`.
- Fixed bugs related to the experimental syntax-based post-processing.

## 1.1.3

### Fixes:

- Fixed a bug that caused the disconnected notification to show up every time when VSCode is started. Now, the notification will only appear after the user modifies the server endpoint settings and the connection fails.

## 1.1.2

### Fixes:

- Fixed a bug that caused the completion to not show up when the completion cache is missing.

## 1.1.0

### Features:

- Added notification with error messages when the connection to the server fails.
- Added a loading status indicator when fetching completions in automatic trigger mode.
- Added a command to display useful online help links, including online documentation, the Tabby Slack community, and the GitHub repository.
- Added an option to mute warning messages for the slow completion response time.
- Updated the config.toml file to include new configuration options, `server.auth` and `completion.timeout`.
- Added experimental features aimed at fine-tuning completion quality. These features are disabled by default but can be enabled by setting the corresponding config flag to `true` in the `config.toml` file, include:
  - `completion.prompt.experimentalStripAutoClosingCharacters`: Strip auto-closing brackets and quotes in prompt suffix, to generate more lines in FIM mode.
  - `postprocess.limitScope.indentation.experimentalKeepBlockScopeWhenCompletingLine`: Use the block scope instead of line scope when using indentation to limit the completion scope and the completion is continuing the current line.
  - `postprocess.limitScope.experimentalSyntax`: Use syntax parser to limit the completion scope.
  - `postprocess.calculateReplaceRange.experimentalSyntax`: Use syntax parser to calculate the completion replace range, to avoid duplicated auto-closing brackets and quotes.

## 1.0.0

### Changes:

- Added auto-closing character check to improve inline completion experience. This improvement ensures proper handling of scenarios involving missing or duplicate closing characters.
- Optimized completion caching for better efficiency, especially when a completion is partially accepted.
- Updated the config.toml template file by removing deprecated options.

## 0.6.1

### Fixes:

- Reduced the frequency of event submissions for anonymous usage tracking.

## 0.6.0

### Features:

- Added manual trigger for inline completion. With the manual trigger mode, you can now use the `Alt + \` hotkey to manually trigger inline completion. This mode can be selected in the extension settings, replacing the old enable/disable inline completion option.
- Improved anonymous usage tracking. Your contribution in sending anonymous usage data is greatly appreciated. However, if you prefer not to participate, you have the option to opt out of this feature within the extension settings.

### Fixes:

- Fixed completion `view` / `select` event logging.
- Adjusted notifications for `highCompletionTimeoutRate` and `slowCompletionResponseTime` to be displayed only once per session.

## 0.5.0

### Incompatible Changes:

- VSCode version requirement is now `≥ 1.82.0`.
- System proxy environment variables are now ignored, including `http_proxy`, `https_proxy`, `all_proxy` and `no_proxy`. Before this change, proxy environment variables are processed, but requests will fail due to lack of supporting for https over http proxy and socks proxy.

### Fixes:

- Fixed a bug that causes auto completion requests cannot be cancelled.

## 0.4.1

### Fixes:

- Updated expired links in the documentation.
- Migrated Tabby cloud authorization tokens and anonymous usage tracking id from the old data directory to the new one.

## 0.4.0

### Features:

- Relocated the user data directory from `$HOME/.tabby/agent` to `$HOME/.tabby-client/agent` to avoid conflicts with Tabby server data. Note that the old data will not be migrated automatically. Please update the config file manually if you have made changes in the old path.
- Added a template config file for Tabby client agent located at `$HOME/.tabby-client/agent/config.toml`.
- Improved code suggestion filtering by indentation context. Suggestions now prioritize completing the current line or logic block, preventing excessively long suggestions.
- Added adaptive completion debouncing for auto completion requests.

### Fixes:

- Resolved conflict with the auto completion widget. The Tabby inline completion will no longer be displayed when the auto completion widget is visible.

## 0.3.0

### Features:

- Added check to see if the editor inline suggestion is enabled. Notifies the user to enable it if it's not.
- Added timeout for auto completion requests. The default timeout is 5 seconds. Added statistics for completion response time and notifies the user if it is too slow.
- Supported setting HTTP request headers in `$HOME/.tabby/agent/config.toml`. The `Authorization` header can be used to set a token for self-hosted servers authorization.

## 0.2.1

### Fixes:

- Set keybindings for accepting inline completion default to VSCode-style, mark Tabby-Style as experimental.

## 0.1.2

### Features:

- Added Tabby extension walkthrough guides.
- Added Tabby-Style keybindings for accepting inline completion as an alternative to VSCode default keybindings.
- Supported reading user config from `$HOME/.tabby/agent/config.toml`, instead of `Developer Options` in extension settings.

### Fixes:

- Improved code suggestion filtering to avoid showing bad suggestions:
  - similar to suffix lines
  - containing repetitive patterns
