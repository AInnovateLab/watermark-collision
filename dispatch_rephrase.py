import hashlib
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from time import sleep
from dispatch_util import find_availabel_gpu, bash_script_header

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")

tmp_dir = "tmp"
dst_dir = 'results'
model_name = 'TheBloke/Llama-2-13B-GPTQ'
dest_dir_name = os.path.join(dst_dir, model_name.split('/')[-1], 'rephrase')
source_dir_name = os.path.join(dst_dir, model_name.split('/')[-1], 'wm')


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
            "--rephraser-file " + self.new_detector,
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
        model_name=model_name,
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
    # KGW-KGW
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW'), True),
    # KGW-KGW-NOWM
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'KGW-KGW-NOWM'), False),
    # KGW-SIR
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR'), True),
    # KGW-SIR-NOWM
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'KGW-SIR-NOWM'), False),
    # KGW-PRW
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW'), True),
    # KGW-PRW-NOWM
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d2_selfhash_t4__g0.25_d2_selfhash_t4.jsonl"), "configs/KGW/g0.25_d2_selfhash_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "KGW/g0.25_d5_selfhash_t4__g0.25_d5_selfhash_t4.jsonl"), "configs/KGW/g0.25_d5_selfhash_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'KGW-PRW-NOWM'), False),
    # SIR-KGW
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW'), True),
    # SIR-KGW-NOWM
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'SIR-KGW-NOWM'), False),
    # SIR-PRW
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW'), True),
    # SIR-PRW-NOWM
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'SIR-PRW-NOWM'), False),
    # SIR-SIR
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR'), True),
    # SIR-SIR-NOWM
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d2_z0_context__w0_g0.5_d2_z0_context.jsonl"), "configs/SIR/w0_g0.5_d2_z0_context.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "SIR/w0_g0.5_d5_z0_context__w0_g0.5_d5_z0_context.jsonl"), "configs/SIR/w0_g0.5_d5_z0_context.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'SIR-SIR-NOWM'), False),
    # PRW-KGW
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW'), True),
    # PRW-KGW-NOWM
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/KGW/g0.25_d2_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/KGW/g0.25_d5_selfhash_t4.yaml", os.path.join(dest_dir_name, 'PRW-KGW-NOWM'), False),
    # PRW-SIR
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR'), True),
    # PRW-SIR-NOWM
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/SIR/w0_g0.5_d2_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/SIR/w0_g0.5_d5_z0_context.yaml", os.path.join(dest_dir_name, 'PRW-SIR-NOWM'), False),
    # PRW-PRW
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW'), True),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW'), True),
    # PRW-PRW-NOWM
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s2.0_t4__f0.5_s2.0_t4.jsonl"), "configs/PRW/f0.5_s2.0_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/PRW/f0.5_s2.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW-NOWM'), False),
    DetectorArg(os.path.join(source_dir_name, "wm/PRW/f0.5_s5.0_t4__f0.5_s5.0_t4.jsonl"), "configs/PRW/f0.5_s5.0_t4.yaml", "configs/PRW/f0.5_s5.0_t4.yaml", os.path.join(dest_dir_name, 'PRW-PRW-NOWM'), False),
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
