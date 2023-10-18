---
authors: [ meng ]

tags: [release]

image: ./twitter-img.jpg
---
# Tabby v0.1.1: Metal inference and StarCoder supports!

We are thrilled to announce the release of Tabby [v0.1.1](https://github.com/TabbyML/tabby/releases/tag/v0.1.1) 👏🏻.

<center>

![Staring tabby riding on llama.cpp](./staring-tabby-on-llama-cpp.png)

*Created with [SDXL-botw](https://huggingface.co/jbilcke-hf/sdxl-botw) and a [twitter post](https://twitter.com/BigCodeProject/status/1684600506658717712) of BigCode*

</center>

Apple M1/M2 Tabby users can now harness Metal inference support on Apple's M1 and M2 chips by using the `--device metal` flag, thanks to [llama.cpp](https://github.com/ggerganov/llama.cpp)'s awesome metal support.

The Tabby team made a [contribution](https://github.com/ggerganov/llama.cpp/pull/3187) by adding support for the StarCoder series models (1B/3B/7B) in llama.cpp, enabling more appropriate model usage on the edge for completion use cases.

<center>

```
llama_print_timings:        load time =   105.15 ms
llama_print_timings:      sample time =     0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =    25.07 ms /     6 tokens (    4.18 ms per token,   239.36 tokens per second)
llama_print_timings:        eval time =   311.80 ms /    28 runs   (   11.14 ms per token,    89.80 tokens per second)
llama_print_timings:       total time =   340.25 ms
```

*Inference benchmarking with [StarCoder-1B](https://huggingface.co/TabbyML/StarCoder-1B) on Apple M2 Max now takes approximately 340ms, compared to the previous time of around 1790ms. This represents a roughly 5x speed improvement.*

</center>


This enhancement leads to a significant inference speed upgrade🚀, for example, It marks a meaningful milestone in Tabby's adoption on Apple devices. Check out our [Model Directory](/docs/models) to discover LLM models with Metal support! 🎁

:::tip
Check out latest Tabby updates on [Linkedin](https://www.linkedin.com/company/tabbyml/) and [Slack community](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)! Our Tabby community is eager for your participation. ❤️ 
:::
