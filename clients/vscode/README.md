# Tabby VSCode Extension

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![VSCode Extension Version](https://img.shields.io/visual-studio-marketplace/v/TabbyML.vscode-tabby)](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)
[![VSCode Extension Installs](https://img.shields.io/visual-studio-marketplace/i/TabbyML.vscode-tabby)](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)
[![Slack Community](https://shields.io/badge/Tabby-Join%20Slack-red?logo=slack)](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)

Tabby is an AI coding assistant that can suggest multi-line code or full functions in real-time.

Tabby VSCode extension is compatible with `VSCode ≥ 1.82.0`, as well as web environments like [vscode.dev](https://vscode.dev).

For more information, please check out our [Website](https://tabbyml.com/) and [GitHub](https://github.com/TabbyML/tabby).
If you encounter any problem or have any suggestion, please [open an issue](https://github.com/TabbyML/tabby/issues/new)!

## Demo

Try our online demo [here](https://tabby.tabbyml.com/playground).

![Demo](https://tabby.tabbyml.com/img/demo.gif)

## Get Started

Once you have installed the Tabby VSCode extension, you can easily get started by following the built-in walkthrough guides. You can access the walkthrough page at any time by using the command `Tabby: Getting Started`.

1. **Setup the Tabby server**: You have two options to set up your Tabby server. You can either build your own self-hosted Tabby server following [this guide](https://tabby.tabbyml.com/docs/installation) or get a Tabby Cloud hosted server [here](https://app.tabbyml.com).  
   **Note**: Tabby Cloud is currently in **closed** beta. Join our [Slack community](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA) and ask in #help-self-hosting channel to get a beta invite.
2. **Connect the extension to your Tabby server**: Use the command `Tabby: Specify API Endpoint of Tabby` to connect the extension to your Tabby server. If you are using a Tabby Cloud server endpoint, follow the instructions provided in the popup messages to complete the authorization process.

Once the setup is complete, Tabby will automatically provide inline suggestions. You can accept the suggestions by simply pressing the `Tab` key.

If you prefer to trigger code completion manually, you can select the manual trigger option in the settings. After that, use the shortcut `Alt + \` to trigger code completion. To access the settings page, use the command `Tabby: Open Settings`.
