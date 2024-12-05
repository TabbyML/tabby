## 1.8.0

### Breaking Changes

- Removed the deprecated `TabbyAgent` interface and related types.

### Features

- Added support for HTTP proxy configuration, defaulting to using HTTP proxy settings in environment variables.
- Included a default git context provider based on the system git command for collecting git repo context.
- Introduced `tabby/status` and `tabby/config` methods, deprecating `tabby/agent` methods.
- Added a method to sync all visible editor ranges for collecting code snippet context to enhance code completion generation.

### Fixes & Improvements

- Added more controls in the initialization options for better compatibility.
- Added a configurable minimal text length threshold to display the completion item.

## 1.7.0

### Breaking Changes

- The tabby-agent will only support running as a language server starting from version 1.7.0.

### Features

- Added support for collecting relative code snippets to enhance code completion.
- Extended the protocol by adding a new method to support inline chat editing.

### Fixes & Improvements

- Fixed a bug that caused unexpected logging output and the generation of an audit.json file in the working directory.

## 1.6.0

### Features

- Added support for multiple choices in inline completion.
- Introduced an experimental feature to generate commit messages.

### Fixes & Improvements

- Improved logging, logging levels can now be set to `silent`, `error`, `info`, `debug` or `verbose`.

## 1.5.0

### Features

- Added support for sending additional context for completion requests, including:
  - filepath
  - git repository information
  - relevant declaration code snippets
  - relevant recently edited code snippets

### Fixes

- Corrected server-side config retrieval behavior for connections to Tabby servers with version < 0.9.

## 1.4.1

### Features

- Added support for loading system-wide CA certificates. Previously, only Node.js bundled CA certificates were used.
- Added support for loading configurations from Tabby server, including `Disabling Client-side Telemetry`.

## 1.3.3

### Features

- Disabled experimental features by default:
  - Stripping auto-closing characters in prompt suffix.
  - Syntax-based code completion scope limit.

## 1.3.2

### Features

- Disabled experimental feature of syntax-based replace range calculation by default.

## 1.3.1

### Features

- Enabled experimental feature of stripping auto-closing characters in prompt suffix by default.
- Enabled experimental feature of syntax-based post-processing by default.
- Removed completion request timeout limit.

### Fixes

- Fixed shebang in cli script to `#!/usr/bin/env node` to support running on macOS. (#1244 - @anoldguy)
- Fixed health checking to be compatibility with Tabby server version 0.2.0 or earlier.

## 1.3.0

The initial version released on npm as a separate package.

### Features

- Added support for run as a language server.
