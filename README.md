<div align="center">

# 🐾 Tabby 🐱
  
[![build status](https://img.shields.io/github/actions/workflow/status/TabbyML/tabby/ci.yml?label=build)](https://github.com/TabbyML/tabby/actions/workflows/ci.yml)
[![Docker pulls](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Slack Community](https://shields.io/badge/Tabby-Join%20Slack-red?logo=slack)](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)

</div>

Tabby is a self-hosted AI coding assistant, offering an open-source and on-premises alternative to GitHub Copilot. It boasts several key features:
* Self-contained, with no need for a DBMS or cloud service.
* OpenAPI interface, easy to integrate with existing infrastructure (e.g Cloud IDE).
* Supports consumer-grade GPUs.

<p align="center">
  <a target="_blank" href="https://tabbyml.github.io/tabby/playground"><img alt="Open in Playground" src="https://img.shields.io/badge/OPEN%20IN%20PLAYGROUND-blue?logo=xcode&style=for-the-badge&logoColor=green"></a>
</p>

<p align="center">
  <img alt="Demo" src="https://user-images.githubusercontent.com/388154/230440226-9bc01d05-9f57-478b-b04d-81184eba14ca.gif">
</p>

## 🔥 What's New
* **09/21/2023** We've hit **10K stars** 🌟 on GitHub! 🚀🎉👏
* **09/18/2023** Apple's M1/M2 Metal inference support has landed in [v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1)!
* **08/31/2023** Tabby's first stable release [v0.0.1](https://github.com/TabbyML/tabby/releases/tag/v0.0.1) 🥳.
<<<<<<< HEAD
=======

<details>
  <summary>Archived</summary>
  
>>>>>>> origin/main
* **08/28/2023** Experimental support for the [CodeLlama 7B](https://github.com/TabbyML/tabby/issues/370).
* **08/24/2023** Tabby is now on [JetBrains Marketplace](https://plugins.jetbrains.com/plugin/22379-tabby)!

</details>

## 👋 Getting Started

The easiest way to start a Tabby server is by using the following Docker command:

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model TabbyML/SantaCoder-1B --device cuda
```
For additional options (e.g inference type, parallelism), please refer to the documentation at https://tabbyml.github.io/tabby.

## 🤝 Contributing

### Get the Code

```bash
git clone --recurse-submodules https://github.com/TabbyML/tabby
cd tabby
```

If you have already cloned the repository, you could run the `git submodule update --recursive --init` command to fetch all submodules.

### Build

1. Set up the Rust environment by following this [tutorial](https://www.rust-lang.org/learn/get-started).

2. Install the required dependencies:
```bash
# For MacOS
brew install protobuf

# For Ubuntu / Debian
apt-get install protobuf-compiler libopenblas-dev
```

3. Now, you can build Tabby by running the command `cargo build`.

### Start Hacking!
... and don't forget to submit a [Pull Request](https://github.com/TabbyML/tabby/compare)


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tabbyml/tabby&type=Date)](https://star-history.com/#tabbyml/tabby&Date)
