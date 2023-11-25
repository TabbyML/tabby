import csv
import json
import pandas as pd

models = ["StarCoder-1B", "StarCoder-3B", "StarCoder-7B", "CodeLlama-7B", "CodeLlama-13B", "WizardCoder-1B", "WizardCoder-3B", "DeepseekCoder-1.3B", "DeepseekCoder-6.7B"]
languages = {"csharp": "C#", "java": "Java", "python": "Python", "typescript": "Typescript"}
files = ["line_completion.jsonl", 'line_completion_rg1_bm25.jsonl', 'line_completion_oracle_bm25.jsonl']
total_records = {'python': 2665, 'java': 2139, 'typescript': 3356, 'csharp': 1768}

headers = ['Model', 'Dataset', 'Records', 'baseline', 'bm25', 'oracle']

stat = []
def get_match(model, language, file):
    count = 0
    with open(f"./data/{model}/{language}/result_{file}") as f:
        for line in f:
            obj = json.loads(line)
            if obj["tabby_eval"]["first_line_matched"]:
                count += 1

    return count

def create_csv():
    for model in models:
        for language in languages.keys():
            x = [model, languages[language], total_records[language]]
            for f in files:
                x.append(get_match(model, language, f))

            stat.append(x)

    df = pd.DataFrame(stat, columns=headers)
    print(df)

    df.to_csv('./tabby_data_pipeline/tabby.csv', index=False)