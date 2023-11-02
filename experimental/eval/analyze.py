import json
import sys

def analyze(model, language, file):
    count = 0
    input_file = "./data/" + model + "/" + language + "/" + file
    output_file = "./data/" + model + "/" + language + "/result_" + file[7:]

    with open(output_file, "w") as fout:
        with open(input_file, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                prompt = obj["prompt"]
                groundtruth = obj["label"]
                prediction = obj["prediction"]
                first_line_groundtruth = groundtruth.split("\n")[0].strip()
                first_line_prediction = prediction.split("\n")[0].strip()

                if first_line_groundtruth == first_line_prediction:
                    match = 1
                    count = count + 1
                else:
                    match = 0
                json.dump(dict(groundtruth=groundtruth, prediction=prediction, first_line_groundtruth=first_line_groundtruth, first_line_prediction=first_line_prediction, match=match), fout)
                fout.write("\n")

    print(str(count) + "records matched!")

if __name__ == "__main__":
    analyze(sys.argv[1], sys.argv[2])