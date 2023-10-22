---
authors: [ gyxlucy ]

tags: [tech design]

---
# Incremental Decoding for Good

In the context of the Transformer model, which is widely used across LLMs, ***decoding*** refers to the process of generating an output sequence from an encoded input. Tabby recenty  [supported ***incremental decoding***](https://github.com/TabbyML/tabby/pull/491) as the default decoding method. This blog will provide an introduction into how Tabby chooses its decoding method for code completion experience 🛠️💡.


## Decoding in Action

Let's say a developer starts to write a list comprehension in Python to get even numbers from a list:

```python
numbers = [1, 2, 3, 4, 5]
evens = [x for x in numbers
```

To simplify the scenario, we assume that the language model maintains a probaility distribution as shown below,

![probability](./probability.png)

Here's how different decoding methods might suggest 🔍:


#### Greedy Decoding 🏆

Greedy decoding selects the most probable next token at each step, which is most intuitive method but can sometimes lead to sub-optimal sequences. This is because it only considers one token at a time and makes such choice greedily. 

In this particular case, greedy decoding would complete the code with `]\n print` as each of the token here has maximum probability given the chosen token before.

![greedy](./greedy.png)

#### Beam Search 🌈
Beam search maintains multiple possible sequences (beams) and expands them, which offers better quality at the expense of higher computation.

Assuming `num_beams=2`, in this case, it'll actually produce ` if x % 2`, as `0.4*0.7=0.28` gives the highest probability when considering a sequence of 2.

![beam](./beam.png)

#### Sampling-based methods 🎲
The two methods above alway produce deterministic results given the language model probability distribution. Often times this isn't the ideal case, especially in conversational use case where users often retry to expect an alternative answer (or think about language translation). Alternatively, sampling-based methods like random sampling, top-k, and top-p sampling introduce randomness to achieve diverse outputs. 

However, as it's now an undeterministic approach, sometimes the models could generate incoherent gibberish results. There are [many different sampling methods](https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/text_generation#transformers.GenerationMixin.sample) to sharp the distribution or redistribute the probability mass to ensure higher chance of generating meaningful tasks. Here we won't go into the details.


#### Incremental Decoding ⏩
You might have realized that there exists a spectrum for decoding methods: simply select the most likely target word in a greedy manner; or search through all possible target sequences and pick the one with highest likelihood. Beam search and sampling-based methods lie in the middle by running approximation using different approaches. 

Beam search might seem like a good compromise between the two extremes, and indeed it provides the correct code completion suggestion in our list comprehension task. However, the search overhead in beam search still **increases linearly** with the length of the decoded sentence, because at each step, we must consider all prefix tokens.

However, if we look closely - it's actually a waste of resource to rerun the decoder for all previous tokens every time. Why don't we reuse the calculation? Indeed, this is exactly what incremental decoding does✅! It maintains a history of previous calculations that are needed for calculating the next outputs. 

![incremental](./incremental.png)

For interested folks, you can refer to Tabby's exact implementation in `IncrementalDecoding` funcion in [`creates/tabby-inference/src/decoding.rs`](https://github.com/TabbyML/tabby/pull/491).

```rust
impl IncrementalDecoding {
    pub fn new(tokenizer: Arc<Tokenizer>, stop_re: Option<Regex>, input_token_ids: &[u32]) -> Self {
        let text = tokenizer
            .decode(input_token_ids, /* skip_special_token = */ true)
            .expect("Cannot decode token from tokenizer.");
        Self {
            tokenizer,
            stop_re,
            token_ids: input_token_ids.to_owned(),
            prefix_offset: 0,
            read_offset: input_token_ids.len(),
            reversed_text: reverse(text),
        }
    }

    pub fn next_token(&mut self, token_id: u32) -> Option<String> {
        let skip_special_token = true;
        self.token_ids.push(token_id);

        let prefix_text = self
            .tokenizer
            .decode(
                &self.token_ids[self.prefix_offset..self.read_offset],
                skip_special_token,
            )
            .expect("Cannot decode token from tokenizer.");

        let new_text = self
            .tokenizer
            .decode(&self.token_ids[self.prefix_offset..], skip_special_token)
            .expect("Cannot decode token from tokenizer.");

        let new_text = if new_text.len() > prefix_text.len() && !new_text.ends_with('�') {
            self.prefix_offset = self.read_offset;
            self.read_offset = self.token_ids.len();
            &new_text[prefix_text.len()..]
        } else {
            ""
        };

        if !new_text.is_empty() {
            self.reversed_text = reverse(new_text) + &self.reversed_text;

            if let Some(re) = &self.stop_re {
                if re.find(&self.reversed_text).is_some() {
                    return None;
                }
            }
        }

        Some(new_text.to_owned())
    }
}
```

Have you found incremental decoding effective? Share your thoughts with us in our [Slack](https://join.slack.com/t/tabbyml/shared_invite/zt-22thejc0z-7ePKeWNCHPX31pEtnT4oYQ) channel 🌍😊!