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
        ]
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
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets.download.download_manager import DownloadConfig

import jsonlines
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Optional, List, Any, Union
from copy import deepcopy

import wm_detector as WMD
import wm_generator as WMG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-config-file", type=str, required=True, help="Yaml file for model and prompt config")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--proxy", type=str, default=None, help="Proxy for downloading models and datasets")
    # Step 1 dataset
    parser.add_argument("--dataset-name", type=str, default="stas/c4-en-10k")
    parser.add_argument("--max-valid", type=int, default=10000, help="Max number of validation samples. Will be ignored if in step 2.")
    parser.add_argument("--valid-only", type=bool, default=False, help="Only log valid result. Will be ignored if in step 2.")
    # Step 2 dataset
    parser.add_argument("--input-file", type=str, help="Path to input file.")
    # Watermark settings
    parser.add_argument("--use-wm", action="store_true", default=False)
    parser.add_argument("--rephraser-file", type=str, default='', help="Yaml file for rephraser.")
    parser.add_argument("--key", type=int, default=2023)
    
    # generate kwargs
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    # I/O
    parser.add_argument("--no-confirm", action="store_true", default=False, help="Overwrite output file without confirmation if set true")
    output_ex_group = parser.add_mutually_exclusive_group(required=True)
    output_ex_group.add_argument("--output-dir", type=str, help="Output directory. If specified, enable automatic naming from the yaml file of generator and detector.",)
    output_ex_group.add_argument("--output-file", type=str, help="Output file name. If specified, disable the automatic naming and ignore the --output-dir setting.",)

    return parser.parse_args()


def truncate_text(s: str, min_len=16, max_len: int = 48):
    s = s.strip().split('.')
    r = []
    l = 0
    for sentence in s:
        sl = len(sentence.strip().split(' '))
        if len(r) > 0 and l > min_len and sl + l > max_len:
            break
        r.append(sentence)
        l += sl
    return '.'.join(r)


class BackTranslate:
    def __init__(self, args) -> None:
        
        """
        Basic configs
        """
        self.args = args
        self.use_wm = args.use_wm
        self.device = args.device
        self.step = args.step
        if self.args.proxy:
            self.proxy = {
                "http": self.args.proxy,
                "https": self.args.proxy,
            }
            self.dataset_proxy = DownloadConfig(proxies=self.proxy)
        else:
            self.proxy = None
            self.dataset_proxy = None
        
        if self.step != 1 and self.args.input_file is None:
            raise ValueError("Input file must be specified in step 2.")
        
        """
        Model & Tokenizer
        """
        self.model_config = OmegaConf.load(args.model_config_file)
        logging.info(f"Loading model and tokenizer: {self.model_config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name, use_fast=True, padding_side="left", proxies=self.proxy)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_name, device_map='auto', trust_remote_code=True, revision="main", proxies=self.proxy)

        """
        Datasets
        """
        if self.step == 1:
            self.prompt: str = self.model_config.model_prompt_translate
            logging.info(f"Loading dataset:{self.args.dataset_name}...")
            self.dataset = load_dataset(self.args.dataset_name, split='train', download_config=self.dataset_proxy)
            # self.dataset = self.dataset["train"]
            logging.info("Preprocessing datasets...")
            self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(truncate_text(b["text"])), return_tensors="pt", truncation=True, max_length=256))
            self.tokenized_dataset.set_format("torch")
        else:
            # Load the generated text if in step 2. Load detector config from jsonl file.
            file_path = self.args.input_file
            logging.info(f"Loading data from: {file_path}")
            with jsonlines.open(file_path, "r") as reader:
                # Use config data to create old detector
                self.orig_config = reader.read()
                wm_data = list(reader)
            data = list({"text": d["texts"][-1], "texts": d["texts"], "results": d["results"]} for d in wm_data)
            self.dataset = Dataset.from_list(data)
            self.prompt = self.model_config.model_prompt_back_translate
            self.tokenized_dataset = self.dataset.map(lambda b: self.tokenizer(self.prompt.format(b["text"]), return_tensors="pt", truncation=True, max_length=256))
            
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
            rephraser_config = OmegaConf.load(args.rephraser_file)
            self.rephraser_config = rephraser_config.generator
            self.rephraser_key = self.args.key
            self.keys.append(self.args.key)
            logging.info(f"Loading Detector config from: {args.rephraser_file}")
            self.detector_configs.append(rephraser_config.detector)
            self.orig_config = {
                "rephraser": [OmegaConf.to_container(rephraser_config)],
                "key": [self.args.key],
                "detector": []
            }
        elif self.step == 2:
            # Load original detector config from jsonl file
            # TODO: Modify here if we need to detect watermark using other detectors
            last_rep_idx = self._get_last_detector_idx()
            self.detector_configs.append(OmegaConf.create(self.orig_config['rephraser'][last_rep_idx]).detector)
            self.keys.append(self.orig_config['key'][last_rep_idx])
            if self.use_wm:
                # TODO: Merge ?
                logging.info(f"Loading new Rephraser & Detector config from: {self.args.rephraser_file}")
                rephraser_config = OmegaConf.load(args.rephraser_file)
                self.detector_configs.append(rephraser_config.detector)
                self.keys.append(self.args.key)
                self.rephraser_config = rephraser_config.generator
                self.rephraser_key = self.args.key
                self.orig_config['rephraser'].append(OmegaConf.to_container(rephraser_config))
                self.orig_config['key'].append(self.args.key)
            else:
                self.orig_config['rephraser'].append(False)
                self.orig_config['key'].append(None)
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
            input_filename_hash = hashlib.md5(
                Path(self.args.input_file).stem.encode("utf-8")
            ).hexdigest()
            input_filename_hash = input_filename_hash[:8]
            rephraser_filename = Path(self.args.rephraser_file).stem
            detector_old_filename = Path(self.args.old_detector_file).stem
            detector_new_filename = Path(self.args.new_detector_file).stem
            file_path = (
                file_path
                / f"{input_filename_hash}@{rephraser_filename}__{detector_old_filename}__{detector_new_filename}.jsonl"
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
                    return
            else:
                logging.info("Overwrite output file due to --no-confirm set")
        logging.info(f"Saving results to {file_path}")
        self.file_path = file_path
        
        # Log all configs to the output file
        with jsonlines.open(file_path, "w") as writer:
            writer.write(self.orig_config)
        
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
        total_num = self.args.max_valid if self.step == 1 else len(self.tokenized_dataset)
        with (
            jsonlines.open(self.file_path, mode="w") as writer,
            tqdm(total=total_num, desc="Valid samples", dynamic_ncols=True) as pbar,
        ):
            """
            TODO: Detector configs here!
            """
            orig_config = deepcopy(self.orig_config)
            orig_config['detector'].append([OmegaConf.to_container(detector) for detector in self.detector_configs])
            writer.write(orig_config)
            valid_num = 0
            stop_id = self.tokenizer.encode('<|eot_id|>')[0]

            for data in self.tokenized_dataset:
                input_ids = data["input_ids"].to(self.device)
                attn_mask = data["attention_mask"].to(self.device)
                output_tokens = self.generator.generate(input_ids, attention_mask=attn_mask, eos_token_id=stop_id, pad_token_id=stop_id, **generate_kwargs)
                if self.use_wm:
                    output_text = self.generator.tokens2text(output_tokens)
                else:
                    # truncate output
                    output_tokens = output_tokens[:, input_ids.size(-1) :]
                    output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                detect_results = []
                for detector in self.detectors:
                    detect_result = detector.detect_tokens(output_tokens)
                    detect_results.append(detect_result.asdict())
                
                # Use the last detector result to validate
                if self.step == 2 or not self.args.valid_only or detect_result.prediction == True or detect_result.prediction is None:
                    valid_num += 1
                    """
                    Save results and generated text to the output file, along with history
                    """
                    if self.step == 1:
                        # Step 1: data['text'] load from dataset, which is a string
                        writer.write(
                            {
                                "results": [detect_results],
                                "texts": [data["text"], output_text],
                            }
                        )
                    else:
                        # Step 2+: history generated texts and detect results load from jsonl file, which is a list
                        det_result = deepcopy(data['results'])
                        texts = deepcopy(data['texts'])
                        det_result.append(detect_results)
                        texts.append(output_text)
                        writer.write(
                            {
                                "results": det_result,
                                "texts": texts,
                            }
                        )
                    pbar.update()

                if self.step == 1 and valid_num >= self.args.max_valid:
                    break
    
        

def main():
    args = parse()
    wm = BackTranslate(args)
    wm.translate()


if __name__ == "__main__":
    main()
