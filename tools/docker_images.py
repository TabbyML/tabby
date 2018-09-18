import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument('--org', default='systran',
                    help='Organization name on Docker Hub.')
parser.add_argument("--version", default="latest",
                    help="Image version.")
parser.add_argument("--push", action="store_true",
                    help="Push the image.")
parser.add_argument("--type", default="all",
                    help="Image type: all, cpu, cuda.")
args = parser.parse_args()

types = []
if args.type in ("cpu", "all"):
    types.append("cpu")
if args.type in ("cuda", "all"):
    types.append("cuda")

for type in types:
    print("Building %s Docker image" % type)
    with open("Dockerfile.in") as dockerfile:
        template = dockerfile.read()

    if type == "cpu":
        build_image = "ubuntu:16.04"
        runtime_image = build_image
        with_cuda = "OFF"
        image_name = "ctranslate2"
    elif type == "cuda":
        build_image = "nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04"
        runtime_image = "nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04"
        with_cuda = "ON"
        image_name = "ctranslate2_gpu"

    template = template.replace("{{BUILD_IMAGE}}", build_image)
    template = template.replace("{{RUNTIME_IMAGE}}", runtime_image)
    template = template.replace("{{WITH_CUDA}}", with_cuda)

    with open("Dockerfile", "w") as dockerfile:
        dockerfile.write(template)

    image = "%s/%s:%s" % (args.org, image_name, args.version)
    subprocess.call(["docker", "build", "-t", image, "-f", "Dockerfile", "."])
    os.remove("Dockerfile")
    if args.push:
        subprocess.call(["docker", "push", image])
