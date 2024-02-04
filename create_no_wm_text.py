import argparse
import logging
from pathlib import Path

import jsonlines
from datasets import Dataset
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wm_detector as WMD
import wm_generator as WMG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


class NoWatermarking:
    def __init__(self, cli_args):
        # read config from generated jsonl
        self.cli_args = cli_args
        logging.info(f"Loading config from: {cli_args.generated_jsonl}")
        with jsonlines.open(cli_args.generated_jsonl) as reader:
            wm_config = reader.read()
            self.args = edict(wm_config)
            self.wm_data = list(reader)

        self.device = cli_args.device
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
        data = list({"text": d["original_text"]} for d in self.wm_data)
        self.dataset = Dataset.from_list(data)

        logging.info("Preprocessing datasets...")

        def tokenize(batch):
            return self.tokenizer(
                batch["text"], return_tensors="pt", truncation=True, max_length=128
            )

        self.tokenized_dataset = self.dataset.map(tokenize)
        self.tokenized_dataset.set_format("torch")

        logging.info(f"Loading Detector config from: {self.args.detector_file}")
        self.detector_config = OmegaConf.load(self.args.detector_file).detector

        self.detector: WMD.WMDetectorBase

        self.init_watermark()

    def init_watermark(self):
        detector_class = WMD.get_detector_class_from_type(self.detector_config.type)
        self.detector = detector_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.args.key,
            **self.detector_config,
        )

    def create_no_wm_text(self):
        """
        Using the LM the continue writing and save the output text.
        """
        input_filepath = Path(self.cli_args.generated_jsonl)
        filepath = input_filepath.with_suffix(".no_wm.jsonl")

        if filepath.exists():
            logging.warning(f"Output file exists: {filepath}")
        logging.info(f"Saving results to {filepath}")
        # generate kwargs
        generate_kwargs = {
            "temperature": self.args.temperature,
            "do_sample": True,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_new_tokens": self.args.max_new_tokens,
            "min_new_tokens": self.args.min_new_tokens,
        }

        with jsonlines.open(filepath, mode="w") as writer:
            writer.write(vars(self.cli_args))

            for data in tqdm(self.tokenized_dataset):
                input_ids = data["input_ids"].to(self.device)
                output_tokens = self.model.generate(input_ids, **generate_kwargs)
                # truncate
                output_tokens = output_tokens[:, input_ids.size(-1) :]

                # output_text = self.generator.tokens2text(output_tokens)
                output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                detect_result = self.detector.detect_tokens(output_tokens)

                writer.write(
                    {
                        "results": detect_result.asdict(),
                        "original_text": data["text"],
                        "generated_text": output_text,
                    }
                )


def parse():
    parser = argparse.ArgumentParser(
        description="Create no-watermarked text from generated watermarked jsonl."
    )
    parser.add_argument("--device", type=str, default="cuda")
    # I/O
    parser.add_argument("generated_jsonl", type=str, help="Generated watermarked jsonl file.")
    return parser.parse_args()


def main():
    args = parse()
    wm = NoWatermarking(args)
    wm.create_no_wm_text()


if __name__ == "__main__":
    main()
