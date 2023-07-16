## 0.1.2

Features:
* Added Tabby extension walkthrough guides.
* Added Tabby-Style keybindings for accepting inline completion as an alternative to VSCode default keybindings.
* Supported reading user config from `$HOME/.tabby/agent/config.toml`, instead of `Developer Options` in extension settings.

Fixes:
* Improved code suggestion filtering to avoid showing bad suggestions:
  * similar to suffix lines
  * containing repetitive patterns