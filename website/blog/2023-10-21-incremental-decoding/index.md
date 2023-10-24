---
authors: [ gyxlucy ]

tags: [tech design]

---
# Incremental Decoding for Good

In the context of the Transformer model, which is widely used across LLMs, ***decoding*** refers to the process of generating an output sequence from an encoded input. Tabby recenty  [implemented ***incremental decoding***](https://github.com/TabbyML/tabby/pull/491) as part of the greedy search. This blog will provide an introduction into why Tabby chooses incremental decoding for code completion tasks 🛠️💡.


## Common Decoding Methods

Here's an example to facilitate understanding different decoding methods:

Let's say a developer starts to write a list comprehension in Python to get even numbers from a list:

```python
numbers = [1, 2, 3, 4, 5]
evens = [x for x in numbers
```

To simplify the scenario, we assume that the language model maintains a probaility distribution as shown below,

![probability](./probability.png)

Here's how different decoding methods might suggest 🔍:



### Beam Search 🌈
Beam search maintains multiple possible sequences (beams) and expands them, which offers better quality at the expense of higher computation.

Assuming `num_beams=2`, in this case, it'll actually produce ` if x % 2`, as `0.4*0.7=0.28` gives the highest probability when considering a sequence of 2.

![beam](./beam.png)

### Greedy Decoding 🏆

Greedy decoding selects the most probable next token at each step, which is most intuitive method but can sometimes lead to sub-optimal sequences. This is because it only considers one token at a time and makes such choice greedily. 

In this particular case, greedy decoding would complete the code with `]\n print` as each of the token here has maximum probability given the chosen token before.

![greedy](./greedy.png)

### Sampling-based methods 🎲
The two methods above alway produce deterministic results given the language model probability distribution. Often times this isn't the ideal case, especially in conversational use case where users often retry to expect an alternative answer (or think about language translation). Alternatively, sampling-based methods like random sampling, top-k, and top-p sampling introduce randomness to achieve diverse outputs. 

However, as it's now an undeterministic approach, sometimes the models could generate incoherent gibberish results. There are [many different sampling methods](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/text_generation#transformers.GenerationMixin.sample) to sharp the distribution or redistribute the probability mass to ensure higher chance of generating meaningful tasks. Here we won't go into the details.

## Era of Streaming for LLM
Latency is key in user experience for many LLM applications. In order to minmize the idle time for users, **streaming response** is commonly adopted. In LLM streaming, we start decoding the response as soon as it's available, instead of waiting for the entier response to be returned. 

Considering streaming process in LLM decoding, although greedy decoding often produces sub-optimal results compared to beam decoding or sampling-methods methods, it wins with its fast and parallelizable computation.

### Incremental Decoding ⏩
However, often times decoding a sequence of tokens one-by-one without considering previous decoded results could produce undesired results. For example,

```
Decoding first token:                ......, 211       ->   "......[ llo]"
Indepently decoding the next token:  ......, 207, 211  ->   "......[ he][ llo]"
```

In the case above, the final decoded string would be `" he llo"` with an undesired space in between. To resolve issues like this, we could cache the already-decoded prefix and append it to the current token to decode together. It is the core idea of **incremental decoding** to take the prefix token into consideration for decoding current tokens. With incremental decoding, we get the desired result for the example above:

```
Incremental decoding:  ......, 207, 211  ->   "......[ hello]"  ✅
```

For interested folks, you can refer to Tabby's exact implementation in `IncrementalDecoding` funcion in [`creates/tabby-inference/src/decoding.rs`](https://github.com/TabbyML/tabby/pull/491).

Have you found incremental decoding effective? Share your thoughts with us in our [Slack](https://join.slack.com/t/tabbyml/shared_invite/zt-22thejc0z-7ePKeWNCHPX31pEtnT4oYQ) channel 🌍😊!