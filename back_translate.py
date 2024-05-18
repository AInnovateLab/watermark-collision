"""
Back translate all in one!
"""

import argparse
import hashlib
import logging
from pathlib import Path
from datasets import load_dataset, Dataset

import jsonlines
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional

import wm_detector as WMD
import wm_generator as WMG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-config-file", type=str, required=True, help="Yaml file for model and prompt config")
    parser.add_argument("--step", type=int, required=True)
    # Step 1 dataset
    parser.add_argument("--dataset-name", type=str, default="stas/c4-en-10k")
    parser.add_argument("--max-valid", type=int, default=10000)
    # Step 2 dataset
    parser.add_argument("--input-file", type=str, required=True, help="Path to input file.")
    # Watermark settings
    parser.add_argument("--use-wm", action="store_true", default=False)
    parser.add_argument("--key", type=int, default=2023)
    # Generator/Detector loading
    parser.add_argument("--rephraser-file", type=str, required=True, help="Yaml file for rephraser.")
    parser.add_argument("--old-detector-file", type=str, required=True, help="Yaml file for old detector.")
    # Dual watermark configuration, only use in step 2
    parser.add_argument("--new-detector-file", type=str, required=True, help="Yaml file for new detector.")
    parser.add_argument("--new-key", type=str, required=True, help="Yaml file for new detector.")
    # generate kwargs
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    # I/O
    parser.add_argument("--no-confirm", action="store_true", default=False, help="Overwrite output file without confirmation if set true")
    output_ex_group = parser.add_mutually_exclusive_group(required=True)
    output_ex_group.add_argument("--output-dir", type=str, help="Output directory. If specified, enable automatic naming from the yaml file of generator and detector.",)
    output_ex_group.add_argument("--output-file", type=str, help="Output file name. If specified, disable the automatic naming and ignore the --output-dir setting.",)
    return parser.parse_args()


class BackTranslate:
    def __init__(self, args) -> None:
        
        """
        Basic configs
        """
        self.use_wm = args.use_wm
        self.device = args.device
        self.step = args.step
        
        """
        Model & Tokenizer
        """
        self.model_config = OmegaConf.load(args.model_config_file)
        logging.info(f"Loading model and tokenizer: {self.args.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name, use_fast=True, padding_side="left")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_name, device_map=0, trust_remote_code=True, revision="main")

        """
        Datasets
        """
        if self.args.step == 1:
            self.prompt: str = self.model_config.model_prompt_translate
            logging.info(f"Loading dataset:{self.args.dataset_name}...")
            self.dataset = load_dataset(self.args.dataset_name)
            self.dataset = self.dataset["train"]
            logging.info("Preprocessing datasets...")
            self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(b["text"]), return_tensors="pt", truncation=True, max_length=128))
            self.tokenized_dataset.set_format("torch")
        else:
            # Load the generated text if in step 2. Load detector config from jsonl file.
            file_path = self.args.input_file
            logging.info(f"Loading data from: {file_path}")
            with jsonlines.open(file_path, "r") as reader:
                # TODO: Use config data to create old detector
                orig_config = reader.read()
                wm_data = list(reader)
            data = list({"text": d["original_text"]} for d in wm_data)
            self.dataset = Dataset.from_list(data)
            self.prompt = self.model_config.model_prompt_back_translate
            self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(b["text"]), return_tensors="pt", truncation=True, max_length=128))
            
        """
        WM Generator & Detector
        """
        if not self.use_wm and self.step == 1:
            raise ValueError("WM Generator must be enabled in step 1.")
        if self.step == 1:
            # Step 1: Load wm generator & detector from args directly
            logging.info(f"Loading Rephraser Generator config from: {args.rephraser_file}")
            self.rephraser_config = OmegaConf.load(self.args.rephraser_file).generator
            logging.info(f"Loading Detector config from: {self.args.old_detector_file}")
            self.detector_config = OmegaConf.load(self.args.old_detector_file).detector
        elif self.step == 2:
            if self.use_wm:
                # TODO: Merge ?
                logging.info(f"Loading new Rephraser & Detector config from: {self.args.new_detector_file}")
                self.detector_config = OmegaConf.create(orig_config['dector_config'])
                self.new_detector_config = OmegaConf.load(self.args.new_detector_file).detector
                self.rephraser_config = OmegaConf.load(self.args.rephraser_file).generator
                self.new_key = self.args.new_key
            else:
                # Use same detector config if no_wm is set
                self.detector_config = self.new_detector_config = OmegaConf.create(orig_config['detector_config'])
        
    def translate(self):
        pass
    
    def back_translate(self):
        pass
        

def main():
    args = parse()
    wm = BackTranslate(args)
    wm.rephrase()


if __name__ == "__main__":
    main()
