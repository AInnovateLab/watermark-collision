import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from time import sleep
from dispatch_util import find_availabel_gpu, bash_script_header

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")

tmp_dir = "tmp"
dst_dir = 'results'
model_name = 'facebook/opt-1.3b'
dir_name = os.path.join(dst_dir, model_name.split('/')[-1], 'wm')


@dataclass
class CreateWMArg:
    generator: str
    output_dir: str
    model: str

    def to_cmd(self):
        args = [
            "python create_wm_text.py",
            "--model-name-or-path " + self.model,
            "--generator-file " + self.generator,
            "--detector-file " + self.generator,
            "--output-dir " + self.output_dir,
        ]
        return " \\\n    ".join(args)

    def __init__(
        self,
        generator_config,
        output_dir,
        model_name="TheBloke/Llama-2-13B-GPTQ",
    ):
        self.generator = generator_config
        self.output_dir = output_dir
        self.model = model_name


arg_list: list[CreateWMArg] = [
    # KGW
    CreateWMArg('configs/KGW/g0.25_d2_selfhash_t4.yaml', os.path.join(dir_name, 'KGW'), model_name),
    CreateWMArg('configs/KGW/g0.25_d5_selfhash_t4.yaml', os.path.join(dir_name, 'KGW'), model_name),
    # PRW
    CreateWMArg('configs/PRW/f0.5_s2.0_t4.yaml', os.path.join(dir_name, 'PRW'), model_name),
    CreateWMArg('configs/PRW/f0.5_s5.0_t4.yaml', os.path.join(dir_name, 'PRW'), model_name),
    # SIR
    CreateWMArg('configs/SIR/w0_g0.5_d2_z0_context.yaml', os.path.join(dir_name, 'SIR'), model_name),
    CreateWMArg('configs/SIR/w0_g0.5_d5_z0_context.yaml', os.path.join(dir_name, 'SIR'), model_name),
]


def executor():
    os.makedirs(tmp_dir, exist_ok=True)
    exec_list = []
    for arg in arg_list:
        while True:
            ava_gpu = find_availabel_gpu(8000)
            if ava_gpu == -1:
                sleep(10)
            else:
                break
        # Write bash script
        script = bash_script_header.format(ava_gpu) + arg.to_cmd()
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
