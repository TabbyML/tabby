## 0.3.0

Features:

- Added check to see if the editor inline suggestion is enabled. Notifies the user to enable it if it's not.
- Added timeout for auto completion requests. The default timeout is 5 seconds. Added statistics for completion response time and notifies the user if it is too slow.
- Supported setting HTTP request headers in `$HOME/.tabby/agent/config.toml`. The `Authorization` header can be used to set a token for self-hosted servers authorization.

## 0.2.1

Fixes:

- Set keybindings for accepting inline completion default to VSCode-style, mark Tabby-Style as experimental.

## 0.1.2

Features:

- Added Tabby extension walkthrough guides.
- Added Tabby-Style keybindings for accepting inline completion as an alternative to VSCode default keybindings.
- Supported reading user config from `$HOME/.tabby/agent/config.toml`, instead of `Developer Options` in extension settings.

Fixes:

- Improved code suggestion filtering to avoid showing bad suggestions:
  - similar to suffix lines
  - containing repetitive patterns
