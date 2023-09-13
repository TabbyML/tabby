---
sidebar_position: 3
---

# Homebrew (Apple M1/M2)
This guide explains how to install Tabby using homebrew.

Thanks to Apple's Accelerate and CoreML frameworks, we can now run Tabby on edge devices with reasonable inference speed. Follow the steps below to set it up using homebrew:

```bash
brew tap TabbyML/tabby
brew install tabby

# Use the flag --HEAD if you're instered in the nightly build.
brew install --HEAD tabby

# Start server with CodeLlama
tabby serve --device cpu --model TabbyML/CodeLlama-7B
```

The compute power of M1/M2 is limited and is likely to be sufficient only for individual usage. If you require a shared instance for a team, we recommend considering Docker hosting with CUDA. You can find more information about Docker [here](./docker).
