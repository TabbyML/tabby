from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

checkpoint = "TabbyML/StarCoder-1B"
device = "mps"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

input_file = "line_completion_rg1_bm25.jsonl"
output_file = "output" + input_file

error_file = "error_" + input_file
ferr = open(error_file, "w")

i = 1
with open(output_file, "w") as fout:
    with open(input_file) as fin:
        for line in fin:
            x = json.loads(line)
            print(str(i) + "begin")
            begin = time.perf_counter()
            prompt = x["prompt"]
            label = x["groundtruth"]
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                outputs = model.generate(inputs, max_new_tokens=20)
                prediction = tokenizer.decode(outputs[0])
                end = time.perf_counter()
                print(f"Time: {end - begin}s")
                print(str(i) + "end")

                json.dump(dict(prompt=prompt, label=label, prediction=prediction), fout)
                fout.write("\n")
            except:
                print(str(i) + "error")
                json.dump(x, ferr)

            i += 1

ferr.close()
