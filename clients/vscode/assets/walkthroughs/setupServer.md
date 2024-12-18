# Setup Tabby Server

The Tabby VSCode extension requires a Tabby server to work, following the instructions below to install and create your account.

## Install Tabby Server

[Tabby](https://www.tabbyml.com/) is an open-source project that supports self-hosting.  
You can choose any of the following methods to install Tabby:

- [Homebrew](https://tabby.tabbyml.com/docs/quick-start/installation/apple/) for macOS with Apple M-series chips.
- [Binary distribution](https://tabby.tabbyml.com/docs/quick-start/installation/windows/) for Windows/Linux users.
  - For NVIDIA GPUs, please check your CUDA version and select the binary distribution with `cuda` version suffix.
  - For other GPUs with Vulkan support, please select the binary distribution with `vulkan` suffix.
- [Docker](https://tabby.tabbyml.com/docs/quick-start/installation/docker/) if you prefer to run Tabby in a container. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is strongly recommended for NVIDIA CUDA support.
- Cloud deployment
  - [Hugging Face Spaces](https://tabby.tabbyml.com/docs/quick-start/installation/hugging-face/)
  - [Modal](https://tabby.tabbyml.com/docs/quick-start/installation/modal/)
  - [SkyPilot](https://tabby.tabbyml.com/docs/quick-start/installation/skypilot/)

## Create Your Account

Visit [http://localhost:8080/](http://localhost:8080/) (or your server address) and follow the instructions to create your account. After creating your account, you can find your token for connecting to the server.

## [Online Supports](command:tabby.openOnlineHelp)

Please refer to our [online documentation](https://tabby.tabbyml.com/docs/) and our [Github repository](https://github.com/tabbyml/tabby) for more information.
If you encounter any problems during server setup, please join our [Slack community](https://links.tabbyml.com/join-slack-extensions) for support or [open an issue](https://github.com/TabbyML/tabby/issues/new/choose).
