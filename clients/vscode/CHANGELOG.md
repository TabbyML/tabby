## 1.20.0

### Features

- **Chat**:
  - You can now use `@` to add a selected file as context for the chat conversation when typing in the chat panel. (Requires connecting to a Tabby server version 0.24.0 or later.)
- Added an inline edit option in the editor right-click context menu.

### Fixes & Improvements

- Retained the chat conversation when moving the chat panel to a different view group. (Requires connecting to a Tabby server version 0.24.0 or later.)
- The chat panel now uses the active editor as context by default.
- Fixed the handling of inline edit cancellations.

## 1.18.0

### Features

- **Chat**:
  - Added support to explicitly select a configured Git repository as the context for chat conversations.
  - Added support to use the active Notebook editor selection as the context for chat conversations.
  - Display error messages and suggest actions when failing to load the chat panel.
  - **Note**: Requires updating the Tabby server to version 0.23.0 or later.

### Fixes & Improvements

- Updated the status bar item to show warning when the server returns an error due to too many requests.
- Improved the UI for the Tabby command palette and when updating the token.

## 1.16.0

### Features

- **Chat**:
  - Symbols referenced in the chat response can now be clicked to navigate to their definitions.
  - **Note**: Requires updating the Tabby server to version 0.21.2, 0.22.0, or later.
- **Code Completion**:
  - Now provides inline completion even when the completion widget is open, based on the selected item in the widget.
  - Automatically adds import statements if needed after accepting a completion that uses a symbol requiring an import.

### Fixes & Improvements

- Streamlined the `Tabby: Connect to Server...` command process and added a server history list for quick selection.
- **Code Completion**: Added a post-processing filter to fix an issue where some code completions contained an extra space in the indentation for certain code completion models.
- **Chat**: Improved the feature to automatically use the active selection code as context.
- **Chat**: Fixed a bug where dragging the chat panel to the right or bottom side of the editor caused it to be blank.
- The HTTP proxy in VSCode settings is no longer used by default. Added an option to enable it.

## 1.14.0

### Features

- Introduced a "Smart Apply" button in the chat panel's generated code block. This feature suggests edits directly in the current editor, enabling users to apply generated code quickly without manual intervention. Note: Requires the latest version of the Tabby server.
- Added a dynamic indicator in the chat panel's input box to show the currently selected text, which will be used as context for the chat conversation. Note: Requires the latest version of the Tabby server.

### Fixes & Improvements

- Resolved an issue where the chat panel's file context navigation failed when the VSCode workspace was not at the root of the git repository containing the target file.
- Fixed a bug where inline completion did not function in the web extension for browsers when opening remote repositories, such as those from GitHub.
- Corrected the storage of user data in the web extension for browsers.

## 1.12.5

### Fixes & Improvements

- Fixed a bug where the inline completion provider was incorrectly registered twice, causing the status bar loading indicator to not correctly show ongoing completion requests.

## 1.12.4

### Fixes & Improvements

- Fixed a bug causing the Tabby web extension to not initialize in browsers. (https://github.com/TabbyML/tabby/issues/3302)

## 1.12.3

### Fixes & Improvements

- Fixed an issue where the chat panel failed to display when the endpoint configuration ended with a trailing slash.
- Improved the context of code completion by adding support for collecting code snippets from recently viewed editors.

## 1.12.2

### Fixes & Improvements

- Fixed a bug that caused the Tabby extension to not initialize when TLS certificates failed to load. (https://github.com/TabbyML/tabby/issues/3248)

## 1.12.1

### Fixes & Improvements

- Fix DataStore initialization that prevented Tabby from starting on a fresh installation. (https://github.com/TabbyML/tabby/issues/3234)

## 1.12.0

### Features

- Updated the chat panel to be compatible with Tabby server v0.18.0 or later.
- Added support to open the chat panel as a tab in editor groups.
- Introduced actions in the quick-fix menu to explain or fix errors using Tabby.

## 1.10.3

### Fixes & Improvements

- Fixed theme detection issue in the chat side panel. (https://github.com/TabbyML/tabby/issues/3031)

## 1.10.2

### Fixes & Improvements

- Enhanced post-processing for generating commit messages to address the issue of quotation marks not being removed in certain scenarios.

## 1.10.1

### Fixes & Improvements

- Enhanced inline editing streaming experience.
- Added the option to utilize `Edit with Tabby` in the quick fix menu for quicker access.
- Supported using the http proxy configuration in the VSCode settings.

## 1.10.0

### Features

- Introduced a new UI for the Tabby command palette, helping you quickly inspect status and navigate to commands. Access the command palette by clicking the `Tabby` label in the status bar.
- Added a new keybinding `Ctrl/Cmd + L` to open the chat panel.
- Enhanced the chat panel conversation context by allowing you to manually add code snippets through right-click menu or `Ctrl/Cmd + L`.
- Added a button in the source control view title to easily generate commit messages.

### Fixes & Improvements

- Renamed the command `Tabby: Edit...` to `Tabby: Start Inline Editing`. You can edit commands from the history now.
- Inline edit streaming steps are now considered as one undo-redo step.
- Fixed a bug where the chat panel did not refresh when server configuration was changed.
- Decreased the delay in navigating to files when clicking on context references in the chat panel.

## 1.8.2

### Features

- Added support for HTTP proxy configuration. Users can now set up an HTTP proxy either through environment variables or in the config file.

### Fixes & Improvements

- Enhanced the chat side panel interaction with various improvements. The chat panel improvements require the latest version of the Tabby server to fully function.
- Updated and optimized the Tabby configurations in the VSCode Settings page.
- Resolved compatibility issue with chat panel on connections to an older version of the Tabby server.
- Users can now manually remove the history of chat edit commands. The maximum number of history entries can be configured in the advanced settings.

## 1.7.4

### Fixes

- Fixed a bug causing the chat view failed to display due to trailing slash in the endpoint config.

## 1.7.3

### Features

- Introducing a new chat view feature that allows users to engage in conversations with their AI assistant. Access the chat view conveniently from the activity bar.
- Introducing a new experimental feature for editing documents using a chat command. Select some text in the editor and press `Ctrl + i` to start.
- Added a set of commands in the command palette to interact with the chat view.

### Fixes & Improvements

- Updated the walkthrough guides.
- Fixed bugs causing the web extension initialization to fail.

## 1.6.3

### Fixes

- Fixed a bug that caused unexpected logging output and the generation of an `audit.json` file in the working directory.

## 1.6.2

### Features

- Added support for multiple choices in inline completion.
- Introduced an experimental feature to generate commit messages.

### Fixes & Improvements

- Improved logging in the VSCode Output channel.
- Fixed a bug causing the chat stream parsing to fail.
- Improved the message extraction when generating commit messages.

## 1.5.4

### Fixes

- Fixed settings title on the VSCode settings page.

## 1.5.3

### Features

- Added support for sending additional context for completion requests, including:
  - Filepath
  - Git repository information
  - Relevant declaration code snippets
  - Relevant recently edited code snippets
- Merged output channels `Tabby` and `Tabby Agent` into one output channel `Tabby`.

### Fixes

- Corrected server-side config retrieval behavior for connections to Tabby servers with version < 0.9.
- Fixed an issue where the indexing worker for recently edited code may cause a stuck.

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
