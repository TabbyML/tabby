<div align="center">

# 🐾 Tabby

[![latest release](https://shields.io/github/v/release/TabbyML/tabby?sort=semver)](https://github.com/TabbyML/tabby/releases/latest)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Docker pulls](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)

[![Slack Community](https://shields.io/badge/Join-Tabby%20Slack-red?logo=slack)](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)
[![Office Hours](https://img.shields.io/badge/Book-Office%20Hours-purple?logo=googlecalendar&logoColor=white)](https://calendly.com/tabby_ml/chat-with-tabbyml)

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

* **12/15/2023** [v0.7.0](https://github.com/TabbyML/tabby/releases/tag/v0.7.0) released with team management and secured access!
* **10/24/2023** ⛳️ Major updates for Tabby IDE plugins across [VSCode/Vim/IntelliJ](https://tabby.tabbyml.com/docs/extensions)!
* **10/15/2023** RAG-based code completion is enabled by detail in [v0.3.0](https://github.com/TabbyML/tabby/releases/tag/v0.3.0)🎉! Check out the [blogpost](https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion/) explaining how Tabby utilizes repo-level context to get even smarter!


<details>
  <summary>Archived</summary>

* **11/27/2023** [v0.6.0](https://github.com/TabbyML/tabby/releases/tag/v0.6.0) released!
* **11/09/2023** [v0.5.5](https://github.com/TabbyML/tabby/releases/tag/v0.5.5) released! With a redesign of UI + performance improvement.
* **10/04/2023** Check out the [model directory](https://tabby.tabbyml.com/docs/models/) for the latest models supported by Tabby.
* **09/18/2023** Apple's M1/M2 Metal inference support has landed in [v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1)!
* **08/31/2023** Tabby's first stable release [v0.0.1](https://github.com/TabbyML/tabby/releases/tag/v0.0.1) 🥳.
* **08/28/2023** Experimental support for the [CodeLlama 7B](https://github.com/TabbyML/tabby/issues/370).
* **08/24/2023** Tabby is now on [JetBrains Marketplace](https://plugins.jetbrains.com/plugin/22379-tabby)!

</details>

## 👋 Getting Started

You can find our documentation [here](https://tabby.tabbyml.com/docs/getting-started).
- 📚 [Installation](https://tabby.tabbyml.com/docs/installation/)
- 💻 [IDE/Editor Extensions](https://tabby.tabbyml.com/docs/extensions/)
- ⚙️ [Configuration](https://tabby.tabbyml.com/docs/configuration)

### Run Tabby in 1 Minute
The easiest way to start a Tabby server is by using the following Docker command:

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model TabbyML/StarCoder-1B --device cuda
```
For additional options (e.g inference type, parallelism), please refer to the [documentation page](https://tabbyml.github.io/tabby).

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

## 🌍 Community
- #️⃣ [Slack](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA) - connect with the TabbyML community 
- 🎤 [Twitter / X](https://twitter.com/Tabby_ML) - engage with TabbyML for all things possible 
- 📚 [LinkedIn](https://www.linkedin.com/company/tabbyml/) - follow for the latest from the community 
- 💌 [Newsletter](https://tinyletter.com/tabbyml/) - subscribe to unlock Tabby insights and secrets



### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tabbyml/tabby&type=Date)](https://star-history.com/#tabbyml/tabby&Date)
