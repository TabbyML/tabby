## 1.3.2

### Fixes:

- Disabled experimental features by default:
  - Stripping auto-closing characters in prompt suffix.
  - Syntax-based code completion scope limit.
  - Syntax-based replace range calculation.

## 1.3.0

### Features:

- Removed the completion request timeout limit. Now, a warning status bar icon will be displayed when the completion request takes too long.
- Enabled the experimental feature of stripping auto-closing characters in the prompt suffix by default.
- Enabled the experimental feature of syntax-based post-processing by default.
- Added support for logging completion dismiss events.

### Fixes:

- Fixed a bug caused `Check Connection` never ends when endpoint config left blank.
- Fixed health checking to be compatible with Tabby server version 0.2.0 or earlier.

## 1.2.0

### Features:

- Added support for partially accepting a completion.
  - Use `Ctrl + Right` to accept the next word, use `Ctrl + Tab` to accept the next line.
  - Keymap scheme can be selected or customized in the plugin settings page.
- Added support for setting Tabby server token in the plugin settings page.
  - You can still configure the token in the agent config file, but the token set in plugin settings page will take precedence.
- A notification will now be displayed when the server requires a token.
- Removed support for automatically opening the authentication page and fetching the token when using Tabby Cloud.
  - To connect to Tabby Cloud server, you need to manually set the token instead. The token already in use will remain usable.

### Fixes:

- Corrected invalid online documentation links.
- Resolved a bug that resulted in empty log files being generated even when the logging level is set to `silent`.
- Fixed bugs related to the experimental syntax-based post-processing.

## 1.1.2

### Fixes:

- Added support for IntelliJ Platform IDEs version 2023.3.x.

## 1.1.1

### Fixes:

- Fix a bug cause the completion does not show up if the completion cache is missing.

## 1.1.0

### Features:

- Changed the default keymap for triggering inline completion to `Ctrl + \`, to avoid conflicts when new UI enabled.
- Added a `Check Connection` button in settings page to check the connection to the server.
- Added notification with error messages when the connection to the server fails.
- Added a loading status indicator when fetching completions in automatic trigger mode.
- Updated the online help links, including online documentation, the Tabby Slack community, and the GitHub repository.
- Added an option to mute warning messages for the slow completion response time.
- Updated the config.toml file to include new configuration options: `server.auth` and `completion.timeout`.
- Added experimental features aimed at fine-tuning completion quality. These features are disabled by default but can be enabled by setting the corresponding config flag to `true` in the `config.toml` file, include:
    - `completion.prompt.experimentalStripAutoClosingCharacters`: Strip auto-closing brackets and quotes in prompt suffix, to generate more lines in FIM mode.
    - `postprocess.limitScope.indentation.experimentalKeepBlockScopeWhenCompletingLine`: Use the block scope instead of line scope when using indentation to limit the completion scope and the completion is continuing the current line.
    - `postprocess.limitScope.experimentalSyntax`: Use syntax parser to limit the completion scope.
    - `postprocess.calculateReplaceRange.experimentalSyntax`: Use syntax parser to calculate the completion replace range, to avoid duplicated auto-closing brackets and quotes.

### Fixes:

- Fixes a bug causing the CJK characters to be rendered incorrectly on Windows.

## 1.0.0

### Changes:

- Added support for completion replacement. If a completion will replace the suffix characters after the cursor, these characters will be hidden.
- Added auto-closing character check to improve inline completion experience. This improvement ensures proper handling of scenarios involving missing or duplicate closing characters.
- Optimized completion caching for better efficiency, especially when a completion is partially accepted.
- Updated the config.toml template file by removing deprecated options.

## 0.6.0

### Features:

- Added an option to switch between automatic/manual completion trigger mode. This option replaces the old enable/disable inline completion option. You can use the `Alt + \` hotkey to trigger inline completion in manual mode.
- Added an option to specify the Node.js binary path.
- Added an action to open online help documents quickly in status bar item menu.

### Fixes:

- Adjusted notifications for `highCompletionTimeoutRate` and `slowCompletionResponseTime` to be displayed only once per session.
- Improved anonymous usage tracking, reduced the frequency of data submissions. Your contribution in sending anonymous usage data is greatly appreciated. However, if you prefer not to participate, you have the option to opt out of this feature within the plugin settings.

## 0.5.0

### Incompatible Changes:

- Node.js version requirement is now v18+.
- System proxy environment variables are now ignored, including `http_proxy`, `https_proxy`, `all_proxy` and `no_proxy`. Before this change, proxy environment variables are processed, but requests will fail due to lack of supporting for https over http proxy and socks proxy.

### Fixes:

- Fixed a bug that causes auto-completion requests cannot be cancelled.
- Migrated Tabby cloud authorization tokens and anonymous usage tracking id from the old data directory to the new one.

## 0.4.0

### Features:

- Relocated the user data directory from `$HOME/.tabby/agent` to `$HOME/.tabby-client/agent` to avoid conflicts with Tabby server data. Note that the old data will not be migrated automatically. Please update the config file manually if you have made changes in the old path.
- Added a template config file for Tabby client agent located at `$HOME/.tabby-client/agent/config.toml`.
- Added check for Node.js installation and notify the user if it is not valid.
- Improved code suggestion filtering by indentation context. Suggestions now prioritize completing the current line or logic block, preventing excessively long suggestions.
- Added adaptive completion debouncing for auto-completion requests.
- Added timeout for auto-completion requests. The default timeout is 5 seconds. Added statistics for completion response time and notifies the user if it is too slow.

### Fixes:

- Fixed a bug that caused the plugin throw errors when initializing without user data file.

## 0.2.0

### Features:

- Added support for Tabby Cloud hosted server authorization.

### Fixes:
- Fixed inlay text rendering issues.

