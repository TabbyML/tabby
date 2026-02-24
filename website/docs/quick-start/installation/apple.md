---
sidebar_position: 3
---

# Homebrew (Apple Silicon M-Series)

This guide explains how to install Tabby using homebrew.

1. Install by homebrew

    ```bash
    brew install tabbyml/tabby/tabby
    ```

2. Start Tabby

    ```bash
    tabby serve
    ```

After Tabby is running, you can access it at [http://localhost:8080](http://localhost:8080).

If you want to host your server on a different port than the default `8080`,
supply the `--port` option. Run `tabby serve --help` to learn about all possible options.
