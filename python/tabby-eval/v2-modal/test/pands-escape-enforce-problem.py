import json

import pandas as pd

if __name__ == "__main__":
    # https://github.com/pandas-dev/pandas/releases
    # latest: 2.2.2, released: 2024-04-11
    # today: 20240717
    print(pd.__version__)

    input_json = '{"text": "//"}'

    df = pd.read_json(json.dumps([json.loads(input_json)]), orient='records')
    output_json = df.to_json(orient="records", lines=True)

    # {"text":"\/\/"}
    print(output_json)
