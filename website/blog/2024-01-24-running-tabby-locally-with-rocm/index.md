---
title: Running Tabby Locally with AMD ROCm
authors: [boxbeam]
tags: [deployment]
---

⚠️ ROCm is only supported in Tabby's nightly builds! Support will be stable in version 0.8.

For those using (compatible) **AMD** graphics cards, you can now run Tabby locally with GPU acceleration using AMD's ROCm toolkit! 🎉

ROCm is AMD's equivalent of NVidia's CUDA library, making it possible to run highly parallelized computations on the GPU. Cuda is open source and supports using multiple GPUs at the same time to perform the same computation.

Currently, Tabby with ROCm is only supported on Linux, and can only be run directly from a compiled binary. In the future, Tabby will be able to run with ROCm on Windows, and we will distribute a Docker container capable of running with ROCm on any platform.

## Install ROCm

Before starting, please make sure you are on a supported system and have ROCm installed. The AMD website [details how to install it](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/install-overview.html), find the instructions for your given platform. Shown below is a successful installation of ROCm packages on Arch Linux.

![ROCm installed on Arch Linux](./rocm-packages.png)

## Install and run Tabby

Once you have installed ROCm, you can [download the precompiled binary for Tabby](https://github.com/TabbyML/tabby/releases/download/nightly/tabby_x86_64-manylinux2014-rocm57) with ROCm, or you can [compile it yourself](https://github.com/TabbyML/tabby/blob/main/CONTRIBUTING.md#local-setup). If compiling yourself, make sure to use the flag `--features rocm` to enable it. ROCm is currently supported in Tabby's nightly builds only, but will be stable with 0.8.8.

Once you have a compiled binary, you can run it with this command:

```
./tabby serve --model TabbyML/StarCoder-1B --device rocm
```

If the command is used correctly and the environment is configured properly, you should see command output similar to the following:  
![Tabby running](./tabby-running.png)  
And enjoy GPU-accelerated code completions! This should be considerably faster than with CPU (I saw a ~5x speedup with StarCoder-1B using a Ryzen 7 5800X and an RX 6950XT).

![Completions demo](./using-completions.png)
