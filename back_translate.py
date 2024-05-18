"""
Back translate all in one!

json config structure:
    {
        "rephraser": [
            {},  // Rephraser config(both generator and detector) if use_wm is set
            False,  // False if use_wm is False
            ...
        ],
        "key": [
            2023,   // Key for watermark generator
            None,   // None if use_wm is False or generator do not need to specify a key
            ...
        ],
    }

"""

import argparse
import hashlib
import logging
from pathlib import Path
from datasets import load_dataset, Dataset

import jsonlines
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, List, Any, Union

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
                # Use config data to create old detector
                self.orig_config = reader.read()
                wm_data = list(reader)
            data = list({"text": d["original_text"]} for d in wm_data)
            self.dataset = Dataset.from_list(data)
            self.prompt = self.model_config.model_prompt_back_translate
            self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(b["text"]), return_tensors="pt", truncation=True, max_length=128))
            
        """
        WM Generator & Detector
        """
        self.detector_configs: List[DictConfig] = []
        self.detectors: List[WMD.WMDetectorBase] = []
        self.keys: List[Union[str, int, None]] = []
        self.generator: Union[WMG.WMGeneratorBase, Any]
        if not self.use_wm and self.step == 1:
            raise ValueError("WM Generator must be enabled in step 1.")
        if self.step == 1:
            # Step 1: Load wm generator & detector from args directly
            logging.info(f"Loading Rephraser Generator config from: {args.rephraser_file}")
            self.rephraser_config = OmegaConf.load(self.args.rephraser_file).generator
            logging.info(f"Loading Detector config from: {self.args.old_detector_file}")
            self.detector_config = OmegaConf.load(self.args.old_detector_file).detector
        elif self.step == 2:
            # Load original detector config from jsonl file
            # TODO: Modify here if we need to detect watermark using other detectors
            self.detector_configs.append(OmegaConf.create(self.orig_config['rephraser'][-1]).detector)
            self.keys.append(self.orig_config['key'][-1])
            if self.use_wm:
                # TODO: Merge ?
                logging.info(f"Loading new Rephraser & Detector config from: {self.args.new_detector_file}")
                self.detector_configs.append(OmegaConf.load(self.args.new_detector_file).detector)
                self.keys.append(self.args.new_key)
                self.rephraser_config = OmegaConf.load(self.args.rephraser_file).generator
                self.rephraser_key = self.args.key
            
                
        """
        Prepare output file
        """
        
    def init_watermark(self):
        if self.use_wm:
            # Load rephraser if use_wm
            generator_class = WMG.get_generator_class_from_type(self.rephraser_config.type)
            self.generator = generator_class(
                model=self.model,
                tokenizer=self.tokenizer,
                key=self.rephraser_key,
                # TODO: generator configs
                **self.generator_config,
            )
        else:
            # no wm, use model directly
            self.generator = self.model
        # Load all detectors in the list
        for detector_config, key in zip(self.detector_configs, self.keys):
            detector_class = WMD.get_detector_class_from_type(detector_config.type)
            self.detectors.append(detector_class(
                model=self.model,
                tokenizer=self.tokenizer,
                key=key,
                **detector_config,
            ))
        
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
