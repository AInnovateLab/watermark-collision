import argparse
import logging
import os
from pathlib import Path

import jsonlines
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wm_detector as WMD
import wm_generator as WMG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


class Watermarking:
    def __init__(self, args):
        # model_name_or_path="TheBloke/Llama-2-7B-GPTQ",
        # dataset_name="stas/c4-en-10k",
        # max_dataset_length=1000
        self.args = args
        self.device = self.args.device
        logging.info(f"Loading model and tokenizer:{self.args.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, use_fast=True, padding_side="left"
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # To use a different branch, change revision
        # For example: revision="main"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path, device_map=0, trust_remote_code=True, revision="main"
        )

        logging.info(f"Loading dataset:{self.args.dataset_name}...")
        self.dataset = load_dataset(self.args.dataset_name)
        self.dataset = self.dataset["train"]

        logging.info("Preprocessing datasets...")

        def tokenize(batch):
            return self.tokenizer(
                batch["text"], return_tensors="pt", truncation=True, max_length=128
            )

        self.tokenized_dataset = self.dataset.map(tokenize)
        self.tokenized_dataset.set_format("torch")

        logging.info(f"Loading Generator config from: {self.args.generator_file}")
        logging.info(f"Loading Detector config from: {self.args.detector_file}")
        self.generator_config = OmegaConf.load(self.args.generator_file).generator
        self.detector_config = OmegaConf.load(self.args.detector_file).detector

        self.generator: WMG.WMGeneratorBase
        self.detector: WMD.WMDetectorBase

        self.init_watermark()

    def init_watermark(self):
        generator_class = WMG.get_generator_class_from_type(self.generator_config.type)
        detector_class = WMD.get_detector_class_from_type(self.detector_config.type)
        self.generator = generator_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.args.key,
            **self.generator_config,
        )
        self.detector = detector_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.args.key,
            **self.detector_config,
        )

    def create_wm_text(self):
        """
        Using the LM the continue writing and save the output text.
        """
        # output I/O
        if self.args.output_file:
            file_path = Path(self.args.output_file)
        elif self.args.output_dir:
            file_path = Path(self.args.output_dir)
            file_path.mkdir(parents=True, exist_ok=True)
            # automatic naming
            generator_filename = Path(self.args.generator_file).stem
            detector_filename = Path(self.args.detector_file).stem
            file_path = file_path / f"{generator_filename}__{detector_filename}.jsonl"
        else:
            raise argparse.ArgumentError(
                None, "Either --output-file or --output-dir must be specified."
            )

        if file_path.exists():
            logging.warning(f"Output file exists: {file_path}")
            override_input = input("Output file exists. Do you want to overwrite? (y/n):")
            if "y" not in override_input.lower():
                logging.info("Aborting.")
                return
        logging.info(f"Saving results to {file_path}")
        # generate kwargs
        generate_kwargs = {
            "truncate_output": True,
            "temperature": self.args.temperature,
            "do_sample": True,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_new_tokens": self.args.max_new_tokens,
            "min_new_tokens": self.args.min_new_tokens,
        }
        generate_kwargs.update(self.generator_config.get("generate_kwargs", {}))

        with jsonlines.open(file_path, mode="w") as writer, tqdm(
            total=self.args.max_valid, desc="Valid samples", dynamic_ncols=True
        ) as pbar:
            writer.write(vars(self.args))
            valid_num = 0

            for data in self.tokenized_dataset:
                input_ids = data["input_ids"].to(self.device)
                output_tokens = self.generator.generate(input_ids, **generate_kwargs)

                output_text = self.generator.tokens2text(output_tokens)
                detect_result = self.detector.detect_tokens(output_tokens)

                if detect_result.prediction == True or detect_result.prediction is None:
                    valid_num += 1
                    writer.write(
                        {
                            "results": detect_result.asdict(),
                            "original_text": data["text"],
                            "generated_text": output_text,
                        }
                    )
                    pbar.update()

                if valid_num >= self.args.max_valid:
                    break


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name-or-path", type=str, default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument("--dataset-name", type=str, default="stas/c4-en-10k")
    parser.add_argument("--max-valid", type=int, default=2000, help="Max number of valid samples")
    # Generator/Detector loading
    parser.add_argument("--generator-file", type=str, required=True, help="Yaml file for generator")
    parser.add_argument("--detector-file", type=str, required=True, help="Yaml file for detector")
    # generate kwargs
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    # Watermark kwargs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--key", type=int, default=2024)
    # I/O
    output_ex_group = parser.add_mutually_exclusive_group(required=True)
    output_ex_group.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. If specified, enable automatic naming from the yaml file of gneerator and detector.",
    )
    output_ex_group.add_argument(
        "--output-file",
        type=str,
        help="Output file name. If specified, disable the automatic naming and ignore the --output-dir setting.",
    )
    return parser.parse_args()


def main():
    args = parse()
    wm = Watermarking(args)
    wm.create_wm_text()


if __name__ == "__main__":
    main()
