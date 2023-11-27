## 1.1.3

### Fixes:

- Fixed a bug that caused the disconnected notification to show up every time when VSCode is started. Now, the notification will only appear after the user modifies the server endpoint settings and the connection fails.

## 1.1.2

### Fixes:

- Fixed a bug that caused the completion to not show up when the completion cache is missing.

## 1.1.0

## Features:

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
