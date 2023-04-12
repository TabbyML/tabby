import json
import os
from argparse import ArgumentParser

from fastapi.openapi.utils import get_openapi

os.environ["DRY_RUN"] = "1"
from tabby.server import app

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        default="docs/openapi.json",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                description=app.description,
                routes=app.routes,
                openapi_version=app.openapi_version,
            ),
            f,
        )
