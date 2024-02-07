import hashlib
import logging
import os
import subprocess
from time import sleep
from dispatch_util import find_availabel_gpu, bash_script_header

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")

tmp_dir = "tmp"
dst_dir = 'results'
model_name = 'facebook/opt-1.3b'
dir_name = os.path.join(dst_dir, model_name.split('/')[-1], 'wm')

arg_list: list[str] = [
    os.path.join(dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"),
    os.path.join(dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"),
    os.path.join(dir_name, "PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"),
    os.path.join(dir_name, "PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"),
    os.path.join(dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"),
    os.path.join(dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl")
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
        script = bash_script_header.format(ava_gpu) + f"""
python create_no_wm_text.py {arg}
"""
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
