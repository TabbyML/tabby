## 1.10.1

### Fixes & Improvements

- Reduced the use of read locks to prevent UI freezing issues when collecting declaration code snippets for code completion.

## 1.10.0

### Features

- **Chat**:
  - Added support to explicitly select a configured Git repository as the context for chat conversations.
  - Added support to use the active editor selection as the context for chat conversations.
  - **Note**: Requires updating the Tabby server to version 0.23.0 or later.

## 1.9.1

### Fixes & Improvements

- Updated the chat panel to be compatible with Tabby server versions 0.21.2, 0.22.0, and later.

## 1.9.0

### Features

- Added a list of actions in the editor's right-click context menu to interact with the Tabby chat panel.

### Fixes & Improvements

- Added support for collecting declaration code snippets to improve the code completion context.
- Fixed the "Test Connection" button in the settings page to wait for the response correctly.
- Fixed the bug where changing the completion trigger mode did not take effect immediately.
- Fixed the chat panel theme syncing issue when switching between light and dark themes.
- Added a help message when failing to create the chat panel.

## 1.8.6

### Fixes & Improvements

- Fixed unhandled exception for requests when the completion API is not available on the server.
- Added support for the latest IntelliJ Platform IDE versions.

## 1.8.4

### Fixes & Improvements

- Fixed an issue where the chat panel failed to display when the endpoint configuration ended with a trailing slash.

## 1.8.3

### Fixes & Improvements

- Fixed a bug that caused the Tabby plugin to not initialize when TLS certificates failed to load. (https://github.com/TabbyML/tabby/issues/3248)

## 1.8.2

### Fixes & Improvements

- Fix DataStore initialization that prevented Tabby from starting on a fresh installation. (https://github.com/TabbyML/tabby/issues/3234)

## 1.8.1

### Features

- Updated the chat panel to compatible with Tabby server v0.18.0 or later.

## 1.7.1

### Features

- Introduced a new chat view feature that allows users to engage in conversations with their AI assistant.
- Added support for HTTP proxy configuration. Users can now set up an HTTP proxy either through environment variables or in the config file.

### Fixes & Improvements

- Fixed a bug where the inline completion service created too many jobs when receiving multiple document change events at the same time, such as during a reformat code action.

## 1.6.3

### Fixes & Improvements

- Fixed a bug that caused the Tabby plugin to get stuck in initialization when an editor has no related virtual file.

## 1.6.2

### Breaking Changes

- The minimum required IDE version has been increased to >= 2023.1.

### Features

- Added support for multiple choices in inline completion. Completion choices can be cycled by shortcuts `Alt + [` and `Alt +]`.
- Added support to collect workspace info and git context to enhance inline completion. Credits to Vladimir (#2044).

### Fixes & Improvements

- Updated the underlay protocol to connect to tabby-agent to use LSP.
- Improved interaction when partially accepting a completion.

## 1.4.1

### Fixes:

- Added support for IntelliJ Platform IDEs version 2024.1.

## 1.4.0

### Features

- Added support for loading system-wide CA certificates. Previously, only Node.js bundled CA certificates were used.
- Added support for loading configurations from Tabby server, including `Disabling Client-side Telemetry`.
- Removed the notification when disconnected from Tabby server, keep only status bar icon.

### Fixes

- Fixed the unexpected behaviors that occur when a closed project is reopened within the same IDE process.

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

