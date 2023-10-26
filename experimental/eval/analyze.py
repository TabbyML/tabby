import json

count = 0

with open("./result.jsonl", "w") as fout:
    with open("./output_starcode_1B_rg1_bm25.jsonl") as fin:
        for line in fin:
            obj = json.loads(line)
            groundtruth = obj["label"]
            prediction = obj["prediction"]
            first_line_groundtruth = groundtruth.split("\n")[0]
            first_line_prediction = prediction.split("\n")[0]
            if first_line_groundtruth == first_line_prediction:
                match = 1
                count = count + 1
            else:
                match = 0
            json.dump(dict(groundtruth=groundtruth, prediction=prediction, first_line_groundtruth=first_line_groundtruth, first_line_prediction=first_line_prediction, match=match), fout)
            fout.write("\n")

print(str(count) + "records matched!")