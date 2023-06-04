import multiprocessing
import sys
import itertools
import subprocess
import os
import argparse

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def csv_print(*args):
    print(*args, sep=',')

N_PHYSICAL_CORES = multiprocessing.cpu_count() // 2
eprint("Optimizing for up to:", N_PHYSICAL_CORES, "cores")

# try best first, ie: closest to the number of cores
configurations = sorted([(i, j) for (i, j) in itertools.product(range(1, N_PHYSICAL_CORES+1), repeat=2) if i*j <= N_PHYSICAL_CORES], key=lambda x: x[0]*x[1], reverse=True)
eprint("Values to explore:", list(configurations))

# do not wait standard input if a src file to translate is given
input_to_translate = bytes() if "--src" in sys.argv else sys.stdin.read().encode()

translate_bin_path = sys.argv[1]

if not os.access(translate_bin_path, os.X_OK):
    eprint("No executable file at:", translate_bin_path)
    exit(1)

def popen_with(inter_threads, intra_threads):
    user_args = [arg for arg in sys.argv[2:] if arg != "--log_profiling"]
    tune_args = ['--inter_threads', str(inter_threads), '--intra_threads', str(intra_threads), '--log_throughput', '--device', 'cpu']
    return subprocess.Popen([translate_bin_path] + user_args + tune_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def performance_with_parameters(inter_threads, intra_threads):
    translate = popen_with(inter_threads, intra_threads)
    translate.stdin.write(input_to_translate)
    translate.stdin.close()
    # any error in the execution of the program will show in stderr and will prevent us to parse tokens/s
    try:
        err = str(translate.stderr.read().decode("utf-8"))
        tokens_per_seconds = float(err)
    except KeyboardInterrupt:
        raise KeyboardInterrupt()
    except:
        eprint(err)
        exit(1)
    memory_used = 0
    # try to get maximum memory used if supported
    try:
        memory_used = os.wait4(translate.pid, 0)[2].ru_maxrss # in MB
    except:
        translate.wait()
    return tokens_per_seconds, memory_used

csv_print("inter_threads", "intra_threads", "tokens/s", "memory_used (in MB)")

try:
    # ignore first launch
    performance_with_parameters(1, N_PHYSICAL_CORES)

    using_less_cores = False
    for configuration in configurations:
        inter_threads = configuration[0]
        intra_threads = configuration[1]

        if inter_threads*intra_threads < N_PHYSICAL_CORES:
            if not using_less_cores:
                using_less_cores = True
                eprint("Now tune with less threads than cores...")

        tokens_per_seconds, memory_used = performance_with_parameters(inter_threads, intra_threads)
        csv_print(inter_threads, intra_threads, tokens_per_seconds, memory_used//1024)
except KeyboardInterrupt:
    eprint()
