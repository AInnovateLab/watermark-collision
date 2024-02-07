import subprocess
import logging
import re

def execute_command(cmd):
    cmd_args = cmd.split()
    out = subprocess.check_output(cmd_args)
    return out.decode("utf-8")


def find_availabel_gpu(required_mem):
    cmd = "nvidia-smi --query-gpu=memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits"
    gpus_info = execute_command(cmd)
    gpus_info = gpus_info.strip().split("\n")
    num_pattern = re.compile(r"(\d+)")
    for idx, gpu_info in enumerate(gpus_info):
        total_mem, free_mem, gpu_used = num_pattern.findall(gpu_info)
        free_mem = int(free_mem)
        if free_mem > required_mem:
            logging.info(f"Found available GPU: GPU ID: {idx}\t Free Mem: {free_mem} MB")
            return idx
    return -1

bash_script_header = """
export HTTP_PROXY=http://localhost:7890
export HTTPS_PROXY=http://localhost:7890
export CUDA_VISIBLE_DEVICES={}

"""