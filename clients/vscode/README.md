# Tabby VSCode Extension

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Extension Version](https://img.shields.io/visual-studio-marketplace/v/TabbyML.vscode-tabby)](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)
[![Visual Studio Marketplace](https://img.shields.io/visual-studio-marketplace/i/TabbyML.vscode-tabby?label=marketplace)](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)
[![Open VSX](https://img.shields.io/open-vsx/dt/TabbyML/vscode-tabby?label=Open-VSX)](https://open-vsx.org/extension/TabbyML/vscode-tabby)
[![Slack Community](https://shields.io/badge/Tabby-Join%20Slack-red?logo=slack)](https://links.tabbyml.com/join-slack)

[Tabby](https://tabby.tabbyml.com/) is an open-source, self-hosted AI coding assistant designed to help you write code more efficiently.

## Installation

The Tabby VSCode extension is available on the [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby) and [Open VSX](https://open-vsx.org/extension/TabbyML/vscode-tabby). To install the extension in VSCode/VSCodium, launch Quick Open (shortcut: `Ctrl/Cmd+P`), paste the following command, and press enter:

```
ext install TabbyML.vscode-tabby
```

## Autocomplete

Tabby suggests multi-line code completions and full functions in real-time as you write code. Give it a try in the [online playground](https://tabby.tabbyml.com/playground).

![Autocomplete Demo](https://tabby.tabbyml.com/img/demo.gif)

## Chat

Tabby can answer general coding questions and specific questions about your codebase with its chat functionality. Here are a few ways to utilize it:

- Start a session in the chat view from the activity bar.
- Select some code and use commands such as `Tabby: Explain This` to ask questions about your selection.
- Request code edits directly by using the `Tabby: Start Inline Editing` command (shortcut: `Ctrl/Cmd+I`).

## Getting Started

1. **Setup Tabby Server**: Set up your self-hosted Tabby server and create your account following [this guide](https://tabby.tabbyml.com/docs/installation).
2. **Connect to Server**: Use the `Tabby: Connect to Server...` command in the command palette and input your Tabby server's endpoint URL and account token. Alternatively, use the [Config File](https://tabby.tabbyml.com/docs/extensions/configurations) for cross-IDE settings.

That's it! You can now start using Tabby in VSCode. Use the `Tabby: Quick Start` command for a detailed interactive walkthrough.

## Additional Resources

- [Online Documentation](https://tabby.tabbyml.com/)
- [GitHub Repository](https://github.com/TabbyML/tabby/): Feel free to [Report Issues](https://github.com/TabbyML/tabby/issues/new/choose) or [Contribute](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md)
- [Slack Community](https://links.tabbyml.com/join-slack): Participate in discussions, seek assistance, and share your insights on Tabby.
