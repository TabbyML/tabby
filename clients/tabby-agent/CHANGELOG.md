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
