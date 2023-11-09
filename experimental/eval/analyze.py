import json
import sys
from eval_utils import postprocess_code_lines, remove_comments
from tree_sitter import Language, Parser

def analyze(model, language, file):

    lang_path = f"build/{language}-lang-parser.so"

    line_match = 0
    statement_match = 0
    parser = Parser()
    if language == "csharp":
        parser_language = Language(lang_path, "c_sharp")
    else:
        parser_language = Language(lang_path, language)
    parser.set_language(parser_language)

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

                tabby_eval["first_statement_prediction"] = postprocess_code_lines(tabby_eval["raw_prompt"], prediction, parser, language)
                tabby_eval["first_statement_groundtruth"] = postprocess_code_lines(tabby_eval["raw_prompt"], groundtruth, parser, language)
                if tabby_eval["first_statement_prediction"] == tabby_eval["first_statement_groundtruth"]:
                    tabby_eval["first_statement_matched"] = True
                    statement_match += 1
                else:
                    tabby_eval["first_statement_matched"] = False

                result["tabby_eval"] = tabby_eval

                json.dump(result, fout)
                fout.write("\n")

    print(f"first line matched: {line_match}")
    print(f"first statement matched: {statement_match}")


analyze(sys.argv[1], sys.argv[2], sys.argv[3])

