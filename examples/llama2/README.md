# Chat with Llama 2

This example demonstrates how to implement an interactive chat session using Llama 2 models.

**The primary goal is to showcase the CTranslate2 usage and API, not the capability of the Llama 2 models nor the best way to manage the context. Some adaptations may be needed to get the best out of these models.**

## Installation

```
pip install -r requirements.txt
```

* The dependencies `transformers` and `torch` are only needed to convert the models.
* To run the model on the GPU, CUDA 11 should be installed on the system (minimally the cuBLAS 11 library)

See also the [CTranslate2 installation documentation](https://opennmt.net/CTranslate2/installation.html).

## Convert the model

The model should be first converted into the CTranslate2 model format. This example converts the 7B model:

```
ct2-transformers-converter --model meta-llama/Llama-2-7b-chat-hf --copy_files tokenizer.model \
    --output_dir llama-2-7b-chat-ct2 --quantization float16 --low_cpu_mem_usage
```

For 8-bit quantization, replace `float16` by `int8_float16`.

**Note:** You need to request an access to the Llama 2 models before you can download them from the Hugging Face Hub. See the instructions on the [model page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). Once you have access to the model, you should login with `huggingface-cli login` before running the conversion command.

## Start a chat session

```
python3 chat.py llama-2-7b-chat-ct2/
```

Simply type your question when the prompt "You:" is displayed.

> Loading the model...
>
> You: Hello, how can you help me?
>
> Llama2: Hello! I'm here to help you in any way I can. Please feel free to ask me any question, and I will do my best to provide a helpful and safe response. Whether you need information on a specific topic, guidance on how to approach a problem, or simply someone to talk to, I'm here to listen and assist.<br/>
> Remember, I'm just an AI and not a human, so I can't provide personal advice or engage in discussions that promote harmful or unethical content. My responses are designed to be socially unbiased and positive in nature, and I will always strive to provide accurate and helpful information.<br/>
> So, what can I help you with today?
>
> You: 

You can also set a system prompt on the command line:

```
python3 chat.py llama-2-7b-chat-ct2/ "Always answer with emojis"
```
