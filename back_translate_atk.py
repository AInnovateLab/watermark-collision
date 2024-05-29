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
        "detector": [
            [
                {}, // Detector configs
                ... // Multiple detectors can be used
            ],
            ...     // Each step will have a list of detector configs, matching the number of score in data
        ],
    }


json data structure:
    {
        "texts": [
            "Original Text",
            "Generated Text 1",
            "Generated Text 2",
            ...
        ],
        "results": [
            [
                {}, // Score Like Object
                ..., // Will have multiple score objects if multiple detectors are used
            ]
        ]
    }
"""

import argparse
import hashlib
import logging
import os.path as osp
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets.download.download_manager import DownloadConfig

import jsonlines
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, List, Any, Union
from copy import deepcopy

import wm_detector as WMD
import wm_generator as WMG

transformers.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-config-file", type=str, required=True, help="Yaml file for model and prompt config")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy for downloading models and datasets")
    # dataset
    parser.add_argument("--input-file", type=str, help="Path to input file.", required=True)
    # Watermark settings
    parser.add_argument("--use-wm", action="store_true", default=False)
    parser.add_argument("--rephraser-file", type=str, default='', help="Yaml file for rephraser.")
    parser.add_argument("--key", type=int, default=2023)
    
    # Original detector settings
    # If not specified, the detector will be loaded from the input file if exists
    parser.add_argument("--old-detector-file", type=str, default='', help="Yaml file for old detector(Optional).")
    parser.add_argument("--old-key", type=str, default='', help="Key for old detector(Optional).")
    
    # generate kwargs
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
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
        self.args = args
        self.use_wm = args.use_wm
        self.device = args.device
        if self.args.proxy:
            self.proxy = {
                "http": self.args.proxy,
                "https": self.args.proxy,
            }
            self.dataset_proxy = DownloadConfig(proxies=self.proxy)
        else:
            self.proxy = None
            self.dataset_proxy = None
        
        """
        Model & Tokenizer
        """
        self.model_config = OmegaConf.load(args.model_config_file)
        logging.info(f"Loading model and tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name, use_fast=True, padding_side="left", proxies=self.proxy)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_name, device_map='auto', trust_remote_code=True, revision="main", proxies=self.proxy)
        self.prompt1 = self.model_config.model_prompt_translate
        self.prompt2 = self.model_config.model_prompt_back_translate
        
        """
        Datasets
        """
        file_path = self.args.input_file
        logging.info(f"Loading data from: {file_path}")
        with jsonlines.open(file_path, "r") as reader:
            self.input_config = reader.read()
            data_lines = list(reader)
        if isinstance(data_lines[0]['results'], dict):
            data_lines = list({'results': [d['results']], 'text': d['generated_text'], 'texts': [{'ori': d['generated_text']}]} for d in data_lines)
        elif isinstance(data_lines[0]['results'], list):
            data_lines = list({'results': d['results'], 'text': d['texts'][-1]['bt'], 'texts': d['texts']} for d in data_lines)
        else:
            logging.error("Invalid input file format: Can't read history results")
            return
        self.dataset = Dataset.from_list(data_lines)
        # self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(b["text"]), return_tensors="pt", truncation=True))
        # self.tokenized_dataset.set_format("torch", columns=['text', 'input_ids', 'attention_mask'], output_all_columns=True)

        """
        WM Generator & Detector
        """
        self.detector_configs: List[DictConfig] = []
        self.detectors: List[WMD.WMDetectorBase] = []
        self.keys: List[Union[str, int, None]] = []
        self.generator: Union[WMG.WMGeneratorBase, Any]
        
        self.orig_config = {
                "rephraser": [],
                "key": [],
                "detector": [],
                "detector_key": [],
            }
        
        # If we can load old detector from the input file, add to the list
        if args.old_detector_file:
            old_detector_config = OmegaConf.load(args.old_detector_file)
            self.detector_configs.append(old_detector_config.detector)
            self.keys.append(args.old_key)
        elif self.input_config.get('detector_file', None):
            # Old format
            detector_file = self.input_config['detector_file']
            if not osp.exists(detector_file):
                ans = input("Old detector file not found, do you want to continue? (y/[n]): ")
                if "y" not in ans.lower():
                    exit(0)
            else:
                old_detector_config = OmegaConf.load(detector_file)
                self.detector_configs.append(old_detector_config.detector)
                self.keys.append(self.input_config.get('key', None))
        elif self.input_config.get('detector', None):
            # New format
            self.orig_config = deepcopy(self.input_config)
            last_rep_idx = self._get_last_detector_idx()
            self.detector_configs.append(OmegaConf.create(self.orig_config['rephraser'][last_rep_idx]).detector)
            self.keys.append(self.orig_config['key'][last_rep_idx])
        else:
            logging.error("No old detector config found, please specify one or check input file format")
            return
        
        if self.use_wm:
            rephraser_config = OmegaConf.load(args.rephraser_file)
            # Generator config
            self.rephraser_config = rephraser_config.generator
            self.rephraser_key = args.key
            # Detector config
            self.detector_configs.append(rephraser_config.detector)
            self.keys.append(args.key)
        self.init_watermark()
            
                
        """
        Prepare output file
        TODO: Need a new auto naming rule
        """
        if self.args.output_file:
            file_path = Path(self.args.output_file)
        elif self.args.output_dir:
            file_path = Path(self.args.output_dir)
            file_path.mkdir(parents=True, exist_ok=True)
            # automatic naming
            input_name = Path(self.args.input_file).stem
            if self.use_wm:
                rephraser_filename = Path(self.args.rephraser_file).stem
                rephraser_type = Path(self.args.rephraser_file).parent.stem
                file_path = (
                    file_path
                    / f"{input_name}#{rephraser_type}_{rephraser_filename}.jsonl"
                )
            else:
                file_path = (
                    file_path
                    / f"{input_name}#NO_WM.jsonl"
                )
        else:
            raise argparse.ArgumentError(
                None, "Either --output-file or --output-dir must be specified."
            )

        if file_path.exists():
            logging.warning(f"Output file exists: {file_path}")
            if not self.args.no_confirm:
                override_input = input("Output file exists. Do you want to overwrite? (y/[n]): ")
                if "y" not in override_input.lower():
                    logging.info("Aborting.")
                    exit(0)
            else:
                logging.info("Overwrite output file due to --no-confirm set")
        logging.info(f"Saving results to {file_path}")
        self.file_path = file_path
        
    def init_watermark(self):
        if self.use_wm:
            # Load rephraser if use_wm
            generator_class = WMG.get_generator_class_from_type(self.rephraser_config.type)
            self.generator = generator_class(
                model=self.model,
                tokenizer=self.tokenizer,
                key=self.rephraser_key,
                **self.rephraser_config,
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
 
    def _get_last_detector_idx(self):
        for i, rephraser_config in enumerate(reversed(self.orig_config['rephraser'])):
            if rephraser_config:
                return i
        
    def translate(self):
        generate_kwargs = {
            "temperature": self.args.temperature,
            "do_sample": True,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_new_tokens": self.args.max_new_tokens,
            "min_new_tokens": self.args.min_new_tokens,
        }
        if self.use_wm:
            # Origin model do not accept these kwargs
            generate_kwargs.update(self.rephraser_config.get("generate_kwargs", {}))
            generate_kwargs.update({"truncate_output": True})
        
        # Run the translation
        total_num = len(self.dataset)
        with (
            jsonlines.open(self.file_path, mode="w", flush='true') as writer,
            tqdm(total=total_num, desc="Valid samples", dynamic_ncols=True) as pbar,
        ):
            """
            TODO: Detector configs here!
            """
            orig_config = deepcopy(self.orig_config)
            orig_config['detector'].append([OmegaConf.to_container(detector) for detector in self.detector_configs])
            orig_config['detector_key'].append(self.keys)
            if self.use_wm:
                orig_config['rephraser'].append(OmegaConf.to_container(self.rephraser_config))
                orig_config['key'].append(self.rephraser_key)
            else:
                orig_config['rephraser'].append(False)
                orig_config['key'].append(None)
            writer.write(orig_config)
            valid_num = 0
            if self.model_config.self_defined_eos:
                stop_id = self.tokenizer.encode(self.model_config.self_defined_eos)[0]
                generate_kwargs.update({"eos_token_id": stop_id, "pad_token_id": stop_id})
                logging.info(f"Using self defined eos token: {self.model_config.self_defined_eos} with id: {stop_id}")
            for datum in self.dataset:
                # Translate
                input_text = self.prompt1.format(datum["text"])
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
                output_tokens = self.generator.generate(input_ids, **generate_kwargs)
                if self.use_wm:
                    output_text = self.generator.tokens2text(output_tokens)
                else:
                    # truncate output
                    output_tokens = output_tokens[:, input_ids.size(-1) :]
                    output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                # Back Translate
                input_text = self.prompt2.format(output_text)
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()
                output_tokens = self.generator.generate(input_ids, **generate_kwargs)
                if self.use_wm:
                    output_text2 = self.generator.tokens2text(output_tokens)
                else:
                    # truncate output
                    output_tokens = output_tokens[:, input_ids.size(-1) :]
                    output_text2 = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                # Detect
                if (len(output_tokens[0]) < self.args.min_new_tokens // 2):
                    # Something strange happened, skip this result
                    print("Warn: Result is skipped due to output is too short")
                    continue
                detect_results = []
                for detector in self.detectors:
                    detect_result = detector.detect_tokens(output_tokens)
                    detect_results.append(detect_result.asdict())
                # Use the last detector result to validate
                
                """
                Save results and generated text to the output file, along with history
                Log results anyway, but only update progress bar if the result is valid if configured
                """
                # Step 2+: history generated texts and detect results load from jsonl file, which is a list
                det_result = deepcopy(datum['results'])
                texts = deepcopy(datum['texts'])
                det_result.append(detect_results)
                texts.append({'t': output_text, 'bt': output_text2})
                writer.write(
                    {
                        "results": det_result,
                        "texts": texts,
                    }
                )
                pbar.update()
    
        

def main():
    args = parse()
    wm = BackTranslate(args)
    wm.translate()


if __name__ == "__main__":
    main()
