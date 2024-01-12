---
sidebar_position: 3
---

# Homebrew (Apple M1/M2)
This guide explains how to install Tabby using homebrew.

Thanks to Apple's Accelerate and CoreML frameworks, we can now run Tabby on edge devices with reasonable inference speed. Follow the steps below to set it up using homebrew:

```bash
brew install tabbyml/tabby/tabby

# Start server with StarCoder-1B
tabby serve --device metal --model TabbyML/StarCoder-1B
```

The compute power of M1/M2 is limited and is likely to be sufficient only for individual usage. If you require a shared instance for a team, we recommend considering Docker hosting with CUDA or ROCm. You can find more information about Docker [here](./docker).
