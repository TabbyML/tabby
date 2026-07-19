---
title: Deploying a Tabby Instance in Hugging Face Spaces
authors:
  - name: Wallace
    url: https://github.com/xxs-wallace
tags: [deployment]
---

### Introduction

[Jetbrains](https://www.jetbrains.com/) JetBrains is a cutting-edge software vendor specializing in the creation of intelligent development tools, including IntelliJ IDEA – the leading Java IDE, RustRover - the best Rust IDE. 

This blog post will guide you through the integration of Tabby, an open-source alternative to GitHub Copilot, which facilitates code completion in RustRover. If you're unfamiliar with Tabby, it's an innovative tool that enhances your coding experience. For more information, visit Tabby on GitHub.

### Getting Started with Tabby

**Prerequisites**
* RustRover IDE.
* Basic knowledge of Node.js and JetBrains' IDE.

**Step 1: Install the Tabby Plugin** Open RustRover and navigate to the marketplace to find and install the Tabby plugin.

**Step 2: Install Node.js** You can follow the instructions on the [Node.js website](https://nodejs.org/en/download/package-manager) to install Node.js. Alternatively, you can use a version manager such as nvm.

**Step 3: Configure Node Binary Path** Open the Tabby plugin settings and specify the path to the Node.js binary. (If the node binary is already accessible via your PATH environment variable, you can skip this step).  For IntelliJ Platform IDEs (Tabby plugin version 0.6.0 or higher):
 * Click on Tabby plugin status bar item and select Open Settings....
 * Enter the path to the node binary on your system in the Node binary field, e.g. /usr/local/bin/node, C:\Program Files\nodejs\node.exe.
 * If you are using a version manager such as nvm, you can enter the path to the node binary installed by the version manager, e.g. ~/.nvm/versions/node/v18.18.0/bin/node.
 * Restart the IDE

**Step 4: Connect to Tabby Server.** You can select Open Settings and set configure `endpoint` with address of tabby server. For example: https://demo.tabbyml.com/. If you do not want to deploy tabby with server mode, you also deploy tabby in your IDE environment. For MacOS:
 * Install tabby simply by `brew install tabbyml/tabby/tabby`. 
 * Start tabby server by `tabby serve --device metal --model StarCoder-1B --chat-model Qwen2-1.5B-Instruct`
 * Set configure `endpoint` with address of local tabby server "http://127.0.0.1:8080/".


#### Verify Tabby is running

Navigate to marketplace of plugins and you can see whether tabby is enabled. And then navigate to tabby plugin settings and you can click `Check connection`.

#### Call code completion API

Now, you are able to call the completion API. When you write some code, RustRover will give code completion suggestion behind, you can apply them by click `command` + `Right Arrow`.
