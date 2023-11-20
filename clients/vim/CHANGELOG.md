## 1.1.1

### Fixes:

- Fix a bug cause the completion does not show up if the completion cache is missing.

## 1.1.0

### Features:

- Updated the config.toml file to include new configuration options: `server.auth` and `completion.timeout`.
- Added a command `:Tabby version` to print the current version of Tabby plugin.
- Added experimental features aimed at fine-tuning completion quality. These features are disabled by default but can be enabled by setting the corresponding config flag to `true` in the `config.toml` file, include:
    - `completion.prompt.experimentalStripAutoClosingCharacters`: Strip auto-closing brackets and quotes in prompt suffix, to generate more lines in FIM mode.
    - `postprocess.limitScope.indentation.experimentalKeepBlockScopeWhenCompletingLine`: Use the block scope instead of line scope when using indentation to limit the completion scope and the completion is continuing the current line.
    - `postprocess.limitScope.experimentalSyntax`: Use syntax parser to limit the completion scope.
    - `postprocess.calculateReplaceRange.experimentalSyntax`: Use syntax parser to calculate the completion replace range, to avoid duplicated auto-closing brackets and quotes.

### Fixes:

- Fixed a bug causing the `<Tab>` key to input unexpected characters when fallback to another plugin script.

## 1.0.2

### Fixes:

- Fixed a bug causing status stuck in 'initializing' when script not found.

## 1.0.1

### Fixes:

- Fixed when resolving the server address, it will now try to prefer to use ipv4 over ipv6. 
- Fixed a bug causing the `<Tab>` key can not fallback to the default behavior.
- Fixed a bug causing the completion replace range is rendered incorrectly.

## 1.0.0

### Initial release
