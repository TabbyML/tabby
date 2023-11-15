import json
import sys
#from eval_utils import postprocess_code_lines, remove_comments
#from tree_sitter import Language, Parser
import pandas as pd

def get_bracket_lang_statement(completion):
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "{", "}"]:
            end_idx = i
            break
    return completion[:end_idx+1] if end_idx else completion


def postprocess_code_lines(prompt, target, language):
    try:
        if language in ["java", "csharp", "typescript"]:
            return get_bracket_lang_statement(target)
        elif language == "python":
            return target.split("\n")[0]
    except Exception as e:
        return target

def analyze(model, language, file):

    line_match = 0
    statement_match = 0

    input_file = f"./data/{model}/{language}/{file}"
    output_file = f"./data/{model}/{language}/result_{file}"

    with open(output_file, 'w') as fout:
        with open(input_file) as fin:
            for line in fin:
                obj = json.loads(line)
                result = {}
                prediction = ""

                for k in obj.keys():
                    if k == "prediction":
                        prediction = str(obj[k])
                        break
                    elif k == "error":
                        break
                    else:
                        result[k] = obj[k]

                tabby_eval = {}
                if file == "line_completion.jsonl":
                    tabby_eval["raw_prompt"] = obj["prompt"]
                else:
                    tabby_eval["raw_prompt"] = obj["crossfile_context"]["text"] + obj["prompt"]

                tabby_eval["prediction"] = prediction

                groundtruth = obj["groundtruth"]
                
                tabby_eval["first_line_prediction"] = prediction.split("\n")[0]
                tabby_eval["first_line_groundtruth"] = groundtruth.split("\n")[0]
                if tabby_eval["first_line_prediction"] == tabby_eval["first_line_groundtruth"]:
                    tabby_eval["first_line_matched"] = True
                    line_match += 1
                else:
                    tabby_eval["first_line_matched"] = False

                tabby_eval["first_statement_prediction"] = postprocess_code_lines(tabby_eval["raw_prompt"], prediction, language)
                tabby_eval["first_statement_groundtruth"] = postprocess_code_lines(tabby_eval["raw_prompt"], groundtruth, language)
                if tabby_eval["first_statement_prediction"] == tabby_eval["first_statement_groundtruth"]:
                    tabby_eval["first_statement_matched"] = True
                    statement_match += 1
                else:
                    tabby_eval["first_statement_matched"] = False

                result["tabby_eval"] = tabby_eval

                json.dump(result, fout)
                fout.write("\n")




