---
sidebar_label: Hugging Face Spaces
---
# Hugging Face Spaces

In this guide, you will learn how to deploy your own Tabby instance and use it for development directly from the Huggingface website.

:::tip
This tutorial is now also available on [Hugging Face](https://huggingface.co/docs/hub/spaces-sdks-docker-tabby)!
:::

## Your first Tabby Space

In this section, you will learn how to deploy a Tabby Space and use it for yourself or your organization.

### Deploy Tabby on Spaces

You can deploy Tabby on Spaces with just a few clicks:

[![Deploy on HF Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/deploy-to-spaces-lg.svg)](https://huggingface.co/spaces/TabbyML/tabby-template-space?duplicate=true)

You need to define the Owner (your personal account or an organization), a Space name, and the Visibility. To secure the api endpoint, we're configuring the visibility as Private.

![Duplicate Space](./duplicate-space.png)

:::tip
If you want to customize the title, emojis, and colors of your space, go to "Files and Versions" and edit the metadata of your README.md file.
:::

You’ll see the Building status and once it becomes Running your space is ready to go. If you don’t see the Tabby swagger UI, try refreshing the page.

![Swagger UI](./swagger-ui.png)

### Your Tabby Space URL
Once Tabby is running, you can use the UI with the <u>Direct URL</u> in the **Embed this Space** option (top right).
You’ll see a URL like this: https://tabbyml-tabby.hf.space. This URL gives you access to a full-screen, stable Tabby instance, and is the API Endpoint for IDE / Editor Extensions to talk with.

### Connect VSCode Extension to Space backend
1. Install the [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby).
2. Open the file located at `~/.tabby-client/agent/config.toml`. Uncomment both the `[server]` section and the `[server.requestHeaders]` section.
   * Set the endpoint to the Direct URL you found in the previous step, which should look something like `https://UserName-SpaceName.hf.space`.
   * As the space is set to **Private**, it is essential to configure the authorization header for accessing the endpoint. You can obtain a token from the [Access Tokens](https://huggingface.co/settings/tokens) page.

<center>

![Agent Config](./agent-config.png)

</center>

3. You'll notice a ✓ icon indicating a successful connection.
![Tabby Connected](./tabby-connected.png)

4. You've complete the setup, now enjoy tabing!

<center>

![Code Completion](./code-completion.png)

</center>

You can also utilize Tabby extensions in other IDEs, such as [JetBrains](https://plugins.jetbrains.com/plugin/22379-tabby).
