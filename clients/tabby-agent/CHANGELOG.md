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
