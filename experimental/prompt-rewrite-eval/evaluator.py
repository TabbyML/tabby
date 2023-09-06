import logging
import os
import random
import requests
import subprocess
import toml

import time
from git import Repo

logging.getLogger().setLevel(logging.INFO)

PORT = 8080
LANGUAGE_SUFFIX_MAP = {
    "python": ["py"],
    "go": ["go"],
    "java": ["java"],
    "javascript_typescript": ["js", "jsx", "mjs", "ts", "tsx", "mts"],
    "rust": ["rs"],
    "php": ["php", "php3", "php4", "php5", "phps", "phpt"],
    "lua": ["lua"]
}

# Handy index class to help get completion requests from sample repository.
class Index():
    # Could use a skiplist to implement the Index.
    # But since typically total numbers of files in a repository
    # is not likely to be a lot, a simple array should suffice.
    def __init__(self):
        self.arr = []
        self.dic = {}
    
    # Need to guarantee the keys in add operations are
    # always monotonously incremental.
    def add(self, key, val):
        self.arr.append(key)
        self.dic[key] = val

    def get(self, key):
        pk = 0
        for k in self.arr:
            if k > key:
                # Return file and the line number in the file
                return self.dic[k], key - pk
            pk = k
        return self.dic[k], key - pk

def wait_for_online(timeout):
    logging.info("Trying to connect to tabby")

    health_url = f"http://127.0.0.1:{PORT}/v1/health"
    
    is_online = False
    till = time.time() + timeout * 1000

    while time.time() < till:
        try:
            r = requests.post(health_url)
            if r.status_code == 200:
                logging.info("Tabby is online now")
                is_online = True
                break
        except:
            logging.info("Retrying to connect")
        time.sleep(1)
    
    return is_online


def index(args):
    binary = args["tabby_path"]
    index_repo_url = args["index_repo_url"]

    # Write to config.toml
    config_file_path = os.path.expanduser("~/.tabby/config.toml")
    config = {
        "repositories": [
            {
                "git_url": index_repo_url,
            }
        ],
        "experimental": {
            "enable_prompt_rewrite": True,
        }
    }
    with open(config_file_path, "w+") as f:
        toml.dump(config, f)

    # Start indexing
    cmd = [binary, "scheduler", "--now"]
    subprocess.run(cmd)

def generate_completion_segments(args):
    sample_repo_url = args["sample_repo_url"]
    language = args["language"]
    prompt_count = args["prompt_count"]

    segments = []

    # Checkout the sample repo
    repo_name = sample_repo_url.split("/")[-1]
    repo_path = os.path.expanduser(repo_name)
    if not os.path.exists(repo_path):
        logging.info("Fetching sample repo")
        Repo.clone_from(sample_repo_url, repo_path)
    
    # Index the files
    sample_file_name = f"tabby_{language}.index"
    sample_file_path = os.path.join(repo_path, sample_file_name)
    if not os.path.exists(sample_file_path):
        logging.info("Traversing the repo to make sample index")
        total_lines = 0
        with open(sample_file_path, "a+") as sample_file_w:
            for root, _, files in os.walk(repo_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    # Exclude the sample file itself
                    if fpath == sample_file_path:
                        continue
                    # Exclude files not in the working language
                    if fname.split(".")[-1] not in LANGUAGE_SUFFIX_MAP[language]:
                        continue
                    with open(fpath, "r") as f:
                        total_lines += len(f.readlines())
                        sample_file_w.write(f'{total_lines} {fpath}\n')

    # Build the index
    # The index file is like:
    #
    # 15 main.py                      (suppose the main.py has 15 loc)
    # 115 src/request.py              (suppose the src/request.py has 100 loc)
    # 265 src/utils.py                (suppose the src/utils has 150 loc)
    # ...
    index = Index()
    with open(sample_file_path, "r") as sample_file_r:
        lines = sample_file_r.readlines()
        for l in lines:
            arr = l.split(" ")
            ln, fpath = arr
            # fpath has "\n" as suffix, trim it
            index.add(int(ln), fpath[:-1])

    for _ in range(prompt_count):
        # Randomly pick a line
        maxline = index.arr[-1]
        line = random.randrange(maxline)
        file_path, line_in_file = index.get(line)
        with open(file_path, "r") as f:
            lines = f.readlines()
            # May overflow but it is okay. Python handles this
            prefix_lines = lines[line_in_file: line_in_file+10]
            suffix_lines = lines[line_in_file+10: line_in_file+20]
            
            prefix = "".join(prefix_lines)
            suffix = "".join(suffix_lines)

            segments.append({
                "prefix": prefix,
                "suffix": suffix
            })

    # Generate query segment
    return segments

def rewrite_prompt(args):
    binary = args["tabby_path"]
    language = args["language"]

    # Start tabby server
    serve_command = [binary, "serve", "--model", "TabbyML/T5P-220M"]
    process = subprocess.Popen(serve_command)

    try:
        # Wait for tabby server to be up online
        if not wait_for_online(5):
            logging.error("Tabby server is not online")
            return
        
        # Generate completion request messages
        completion_url = f"http://127.0.0.1:{PORT}/v1/completions"
        segments = generate_completion_segments(args)
        for s in segments:
            req = {
                "language": language,
                "segments": s,
            }

            r = requests.post(completion_url, json=req)
            logging.info(r.status_code)
    finally:
        process.terminate()

def main():
    args = toml.load("eval.toml")
    index(args)
    rewrite_prompt(args)

if __name__ == "__main__":
    main()