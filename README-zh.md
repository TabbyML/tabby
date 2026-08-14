<div align="center">

# 🐾 Tabby

[📚 文档](https://tabby.tabbyml.com/docs/welcome/) • [💬 Slack](https://links.tabbyml.com/join-slack) • [🗺️ 路线图](https://tabby.tabbyml.com/docs/roadmap/)

[![最新版本](https://shields.io/github/v/release/TabbyML/tabby)](https://github.com/TabbyML/tabby/releases/latest)
[![欢迎 PR](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)
[![Docker 下载量](https://img.shields.io/docker/pulls/tabbyml/tabby)](https://hub.docker.com/r/tabbyml/tabby)
[![代码覆盖率](https://codecov.io/gh/TabbyML/tabby/graph/badge.svg?token=WYVVH8MKK3)](https://codecov.io/gh/TabbyML/tabby)

[English](/README.md) |
[简体中文](/README-zh.md) |
[日本語](/README-ja.md)

</div>

Tabby 是一个自托管的 AI 编码助手，提供 GitHub Copilot 的开源和本地替代方案。它具有以下几个关键特性：
* 独立运行，无需 DBMS 或云服务。
* OpenAPI 接口，易于与现有基础设施集成（例如云 IDE）。
* 支持消费级 GPU。

<p align="center">
  <a target="_blank" href="https://tabby.tabbyml.com"><img alt="打开在线演示" src="https://img.shields.io/badge/OPEN_LIVE_DEMO-blue?logo=xcode&style=for-the-badge&logoColor=green"></a>
</p>

<p align="center">
  <img alt="演示" src="https://user-images.githubusercontent.com/388154/230440226-9bc01d05-9f57-478b-b04d-81184eba14ca.gif">
</p>

## 🔥 最新动态
* **2025/03/31** [v0.27](https://github.com/TabbyML/tabby/releases/tag/v0.27.0) 发布，聊天侧边栏中新增更丰富的 `@` 菜单。
* **2025/02/05** Tabby [v0.24.0](https://github.com/TabbyML/tabby/releases/tag/v0.24.0) 中引入 LDAP 认证和更好的后台作业通知！✨
* **2025/02/04** [VSCode 1.20.0](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog) 升级！可以通过 @ 提及文件将其添加为聊天上下文，并通过新的右键选项进行内联编辑！
* **2025/01/10** Tabby [v0.23.0](https://github.com/TabbyML/tabby/releases/tag/v0.23.0) 提供增强的代码浏览体验和聊天侧边栏改进！

<details>
  <summary>存档</summary>
* **2024/12/24** 在 Tabby [v0.22.0](https://github.com/TabbyML/tabby/releases/tag/v0.22.0) 中引入 **通知框**！
* **2024/12/06** Tabby [v0.21.0](https://github.com/TabbyML/tabby/releases/tag/v0.21.0) 中即将推出 Llamafile 部署集成和增强的答案引擎用户体验！🚀
* **2024/11/10** 在 Tabby [v0.20.0](https://github.com/TabbyML/tabby/releases/tag/v0.20.0) 中，答案引擎支持在不同的后端聊天模型之间切换！
* **2024/10/30** Tabby [v0.19.0](https://github.com/TabbyML/tabby/releases/tag/v0.19.0) 在主页上展示最近共享的线程，以提高其可发现性。
* **2024/07/09** 🎉宣布 [Tabby 中的 Codestral 集成](https://tabby.tabbyml.com/blog/2024/07/09/tabby-codestral/)！
* **2024/07/05** Tabby [v0.13.0](https://github.com/TabbyML/tabby/releases/tag/v0.13.0) 引入了 ***答案引擎***，这是一个面向内部工程团队的中央知识引擎。它与开发团队的内部数据无缝集成，提供可靠和精确的答案以增强开发人员的能力。
* **2024/06/13** [VSCode 1.7](https://marketplace.visualstudio.com/items/TabbyML.vscode-tabby/changelog) 标志着一个重要的里程碑，提供了贯穿整个编码体验的多功能聊天体验。来试试最新的 **侧边栏聊天** 和 **通过聊天命令编辑**！
* **2024/06/10** 最新 📃博客文章发布，关于 Tabby 中 [增强的代码上下文理解](https://tabby.tabbyml.com/blog/2024/06/11/rank-fusion-in-tabby-code-completion/)！
* **2024/06/06** Tabby [v0.12.0](https://github.com/TabbyML/tabby/releases/tag/v0.12.0) 发布，带来 🔗**无缝集成**（Gitlab SSO，自托管 GitHub/GitLab 等），到 ⚙️**灵活配置**（HTTP API 集成）和 🌐**扩展功能**（代码浏览器中的仓库上下文）！
* **2024/05/22** Tabby [VSCode 1.6](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby) 提供 **多种选择** 的内联补全和 **自动生成的提交信息**🐱💻！
* **2024/05/11** [v0.11.0](https://github.com/TabbyML/tabby/releases/tag/v0.11.0) 带来了重要的企业升级，包括 📊**存储使用**统计，🔗**GitHub & GitLab** 集成，📋**活动**页面，以及期待已久的 🤖**询问 Tabby** 功能！
* **2024/04/22** [v0.10.0](https://github.com/TabbyML/tabby/releases/tag/v0.10.0) 发布，推出最新的 **报告** 标签，提供团队使用 Tabby 的分析。
* **2024/04/19** 📣 Tabby 现在结合了 [本地相关代码片段](https://github.com/TabbyML/tabby/pull/1844)（来自本地 LSP 的声明和最近修改的代码）用于代码补全！
* **2024/04/17** CodeGemma 和 CodeQwen 模型系列现已添加到 [官方注册表](https://tabby.tabbyml.com/docs/models/)！
* **2024/03/20** [v0.9](https://github.com/TabbyML/tabby/releases/tag/v0.9.1) 发布，重点推出完整功能的管理员 UI。
* **2023/12/23** 通过 [SkyServe](https://skypilot.readthedocs.io/en/latest/serving/sky-serve.html) 🛫 从 SkyPilot 无缝 [在任何云上部署 Tabby](https://tabby.tabbyml.com/docs/installation/skypilot/)。
* **2023/12/15** [v0.7.0](https://github.com/TabbyML/tabby/releases/tag/v0.7.0) 发布，带有团队管理和安全访问！
* **2023/10/15** 在 [v0.3.0](https://github.com/TabbyML/tabby/releases/tag/v0.3.0) 中启用了基于 RAG 的代码补全🎉！查看 [博客文章](https://tabby.tabbyml.com/blog/2023/10/16/repository-context-for-code-completion/) 了解 Tabby 如何利用仓库级上下文变得更智能！
* **2023/11/27** [v0.6.0](https://github.com/TabbyML/tabby/releases/tag/v0.6.0) 发布！
* **2023/11/09** [v0.5.5](https://github.com/TabbyML/tabby/releases/tag/v0.5.5) 发布！UI 重新设计 + 性能改进。
* **2023/10/24** ⛳️ Tabby IDE 插件的重大更新，适用于 [VSCode/Vim/IntelliJ](https://tabby.tabbyml.com/docs/extensions)！
* **2023/10/04** 查看 [模型目录](https://tabby.tabbyml.com/docs/models/) 了解 Tabby 支持的最新模型。
* **2023/09/18** 苹果 M1/M2 Metal 推理支持已在 [v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1) 中推出！
* **2023/08/31** Tabby 的第一个稳定版本 [v0.0.1](https://github.com/TabbyML/tabby/releases/tag/v0.0.1) 🥳。
* **2023/08/28** 对 [CodeLlama 7B](https://github.com/TabbyML/tabby/issues/370) 的实验性支持。
* **2023/08/24** Tabby 现已在 [JetBrains Marketplace](https://plugins.jetbrains.com/plugin/22379-tabby) 上架！

</details>

## 👋 入门指南

您可以在[这里](https://tabby.tabbyml.com/docs/getting-started)找到我们的文档。
- 📚 [安装](https://tabby.tabbyml.com/docs/installation/)
- 💻 [IDE/编辑器扩展](https://tabby.tabbyml.com/docs/extensions/)
- ⚙️ [配置](https://tabby.tabbyml.com/docs/configuration)

### 1 分钟运行 Tabby
启动 Tabby 服务器的最简单方法是使用以下 Docker 命令：

```bash
docker run -it \
  --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby \
  serve --model StarCoder-1B --device cuda --chat-model Qwen2-1.5B-Instruct
```
有关其他选项（例如推理类型、并行性），请参阅[文档页面](https://tabbyml.github.io/tabby)。

## 🤝 贡献

完整指南请参见 [CONTRIBUTING.md](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md);

### 获取代码

```bash
git clone --recurse-submodules https://github.com/TabbyML/tabby
cd tabby
```

如果您已经克隆了仓库，可以运行 `git submodule update --recursive --init` 命令来获取所有子模块。

### 构建

1. 按照此 [教程](https://www.rust-lang.org/learn/get-started) 设置 Rust 环境。

2. 安装所需的依赖项：
```bash
# 对于 MacOS
brew install protobuf

# 对于 Ubuntu / Debian
apt install protobuf-compiler libopenblas-dev
```

3. 安装有用的工具：
```bash
# 对于 Ubuntu
apt install make sqlite3 graphviz
```

4. 现在，您可以通过运行命令 `cargo build` 来构建 Tabby。

### 开始开发！
... 别忘了提交一个 [Pull Request](https://github.com/TabbyML/tabby/compare)

## 🌍 社区
- 🎤 [Twitter / X](https://twitter.com/Tabby_ML) - 与 TabbyML 互动，探索所有可能性
- 📚 [LinkedIn](https://www.linkedin.com/company/tabbyml/) - 关注社区的最新动态
- 💌 [新闻通讯](https://newsletter.tabbyml.com/archive) - 订阅以解锁 Tabby 的见解和秘密

### 🔆 活动

![Git 仓库活动](https://repobeats.axiom.co/api/embed/e4ef0fbd12e586ef9ea7d72d1fb4f5c5b88d78d5.svg "Repobeats 分析图")

### 🌟 星标历史

[![星标历史图](https://star-history.dera.page/svg?repos=tabbyml/tabby&type=Date)](https://star-history.dera.page/#tabbyml/tabby&Date)
