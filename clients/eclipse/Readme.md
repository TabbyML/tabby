# Tabby Plugin for Eclipse IDE
Tabby is an AI coding assistant that can suggest multi-line code or full functions in real-time.

Tabby IntelliJ Platform plugin works with all [Eclipse IDEs](https://www.eclipse.org/ide) that have 2023-09 or later versions.

## Getting Started

1. Set up the Tabby Server: you can build your self-hosted Tabby server following [this guide](https://tabby.tabbyml.com/docs/installation/), or get a Tabby Cloud hosted server [here](https://app.tabbyml.com).  
  **Note**: Tabby Cloud is currently in **closed** beta. Join our [Slack community](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA) and ask in Tabby Cloud channel to get a beta invite.
1. Install Tabby plugin from this project by running it via eclipse
2. Install [Node.js](https://nodejs.org/en/download/) version 18.0 or higher.
3. Open the eclipse settings and select TabbyML Section 
   1. Fill in the server endpoint URL to connect the plugin to your Tabby server.  
     * If you are using default port `http://localhost:8080`, you can skip this step.  
     * If you are using a Tabby Cloud server endpoint, follow the instructions provided in the popup messages to complete the authorization process. 

## Troubleshooting

If you encounter any problem, please check out our [troubleshooting guide](https://tabby.tabbyml.com/docs/extensions/troubleshooting).

## Development and Build

To develop and build Tabby eclipse plugin, please clone [this directory](https://github.com/TabbyML/tabby/tree/main/clients/eclipse) and import it into Eclipse IDE (Preferrably the [Package for Eclipse Committers](https://www.eclipse.org/downloads/packages/release/2023-09/r/eclipse-ide-eclipse-committers)).

