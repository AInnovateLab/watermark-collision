import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from time import sleep

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


bash_script = """
export HTTP_PROXY=http://localhost:7890
export HTTPS_PROXY=http://localhost:7890
export CUDA_VISIBLE_DEVICES={}

"""

tmp_dir = "tmp"


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


@dataclass
class DetectorArg:
    source_file: str
    old_detector: str
    new_detector: str
    output_dir: str
    use_wm: bool
    key: str
    model: str

    def to_cmd(self):
        args = [
            "python rephrase.py",
            "--model-name-or-path " + self.model,
            "--new-key " + str(self.key),
            "--input-file " + self.source_file,
            "--rephraser-file " + self.old_detector,
            "--old-detector-file " + self.old_detector,
            "--new-detector-file " + self.new_detector,
            "--output-dir " + self.output_dir,
        ]
        if self.use_wm:
            args.append("--use-wm")
        args.append("--no-confirm")
        return " \\\n    ".join(args)

    def __init__(
        self,
        source_file,
        old_detector,
        new_detector,
        output_dir,
        use_wm,
        model_name="TheBloke/Llama-2-13B-GPTQ",
        new_key=2023,
    ):
        self.source_file = source_file
        self.old_detector = old_detector
        self.new_detector = new_detector
        self.output_dir = output_dir
        self.use_wm = use_wm
        self.model = model_name
        self.key = new_key


arg_list: list[DetectorArg] = [
    # KGW-KGW NO WM
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "results/KGW-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "results/KGW-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "results/KGW-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "results/KGW-KGW-NOWM",
        False,
    ),
    # KGW-SIR
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "results/KGW-SIR",
        True,
    ),
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "results/KGW-SIR",
        True,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "results/KGW-SIR",
        True,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "results/KGW-SIR",
        True,
    ),
    # KGW-SIR NO WM
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "results/KGW-SIR-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "results/KGW-SIR-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "results/KGW-SIR-NOWM",
        False,
    ),
    DetectorArg(
        "results/kgw/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "results/KGW-SIR-NOWM",
        False,
    ),
    # SIR-KGW NO WM
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "results/SIR-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "results/SIR-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/KGW/g0.25_d2_selfhash_t4.yaml",
        "results/SIR-KGW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/KGW/g0.25_d5_selfhash_t4.yaml",
        "results/SIR-KGW-NOWM",
        False,
    ),
    # SIR-PRW
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/PRW/f0.5_s2.0_t4.yaml",
        "results/SIR-PRW",
        True,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/PRW/f0.5_s5.0_t4.yaml",
        "results/SIR-PRW",
        True,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/PRW/f0.5_s2.0_t4.yaml",
        "results/SIR-PRW",
        True,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/PRW/f0.5_s5.0_t4.yaml",
        "results/SIR-PRW",
        True,
    ),
    # SIR-PRW NO WM
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/PRW/f0.5_s2.0_t4.yaml",
        "results/SIR-PRW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d2_z0_context.yaml",
        "configs/PRW/f0.5_s5.0_t4.yaml",
        "results/SIR-PRW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/PRW/f0.5_s2.0_t4.yaml",
        "results/SIR-PRW-NOWM",
        False,
    ),
    DetectorArg(
        "results/sir/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl",
        "configs/SIR/w0_g0.5_d5_z0_context.yaml",
        "configs/PRW/f0.5_s5.0_t4.yaml",
        "results/SIR-PRW-NOWM",
        False,
    ),
]


def executor():
    os.makedirs(tmp_dir, exist_ok=True)
    exec_list = []
    for arg in arg_list:
        while True:
            ava_gpu = find_availabel_gpu(12000)
            if ava_gpu == -1:
                sleep(10)
            else:
                break
        # Write bash script
        script = bash_script.format(ava_gpu) + arg.to_cmd()
        fn = os.path.join(tmp_dir, hashlib.md5(script.encode()).hexdigest() + ".sh")
        log_fn = fn[:-2] + "log"
        logging.info("Writing command to " + fn)
        logging.info(script)
        with open(fn, "w") as f:
            f.write(script)
        log_f = open(log_fn, mode="w")
        logging.info("Running bash script")
        l = subprocess.Popen(f"bash {fn}", shell=True, stdout=log_f, stderr=log_f)
        exec_list.append((l, log_f, fn))
        logging.info("Sleep for 60 secs to wait for program launching...")
        sleep(60)
    logging.info("Waiting subprocess exit...")
    for p, f, cmd in exec_list:
        p.wait()
        if p.returncode != 0:
            logging.warning(f"Error occured while executing {cmd}")
    logging.info("See you")


if __name__ == "__main__":
    executor()
