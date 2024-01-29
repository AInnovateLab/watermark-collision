import argparse
import os
import pathlib

import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    LogitsProcessorList,
)

import wm_detector as WMD
import wm_generator as WMG
from watermarking.extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor


class Watermarking:
    def __init__(self, args):
        # model_name_or_path="TheBloke/Llama-2-7B-GPTQ",
        # dataset_name="stas/c4-en-10k",
        # max_dataset_length=1000
        self.args = args
        self.device = self.args.device
        print(f"Loading model and tokenizer:{self.args.model_name_or_path}...", end="")
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
        print("Done!")

        print(f"Loading dataset:{self.args.dataset_name}...", end="")
        self.dataset = load_dataset(self.args.dataset_name)
        print("Done!")
        self.dataset = self.dataset["train"]

        def tokenize(batch):
            return self.tokenizer(
                batch["text"], return_tensors="pt", truncation=True, max_length=128
            )

        self.tokenized_dataset = self.dataset.map(tokenize)
        self.tokenized_dataset.set_format("torch")

        self.generator: WMG.WMGeneratorBase
        self.detector_origin: WMD.WMDetectorBase
        self.detector_new: WMD.WMDetectorBase

        self.init_watermark()

    def init_watermark(self):
        if self.args.watermark_name == "KGW":
            self.generator = WMG.KGWWMGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                gamma=self.args.gamma,
                delta=self.args.delta,
                seeding_scheme=self.args.seeding_scheme,
                key=self.args.hash_key,
            )
            self.detector = WMD.KGWWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                gamma=self.args.gamma,  # should match original setting
                seeding_scheme=self.args.seeding_scheme,  # should match original setting
                key=self.args.hash_key,
                z_threshold=4.0,
            )

        elif self.args.watermark_name == "SIR":
            self.generator = WMG.SIRWMGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                window_size=0,
                gamma=self.args.gamma,
                delta=self.args.delta,
                key=self.args.hash_key,
            )
            self.detector = WMD.SIRWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                window_size=0,
                gamma=self.args.gamma,
                delta=self.args.delta,
                key=self.args.hash_key,
                z_threshold=0,
            )

    def create_wm_text(self):
        """
        Using the LM the continue writing and save the output text.
        """
        os.makedirs(self.args.output_dir, exist_ok=True)

        file_path = pathlib.Path(self.args.output_dir) / self.args.output_file

        with jsonlines.open(file_path, mode="w") as writer, tqdm(
            total=self.args.max_valid, desc="Valid samples"
        ) as pbar:
            args_dict = vars(self.args)
            writer.write(args_dict)
            valid_num = 0

            for data in self.tokenized_dataset:
                input_ids = data["input_ids"].to(self.device)
                # print(input_ids)
                output_tokens = self.generator.generate(
                    input_ids,
                    truncate_output=True,
                    temperature=self.args.temperature,
                    do_sample=True,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    max_new_tokens=self.args.max_new_tokens,
                    min_new_tokens=self.args.min_new_tokens,
                )

                # if decoder only model, then we need to isolate the
                # newly generated tokens as only those are watermarked, the input/prompt is not

                output_text = self.generator.tokens2text(output_tokens)
                detect_result = self.detector.detect_tokens(output_tokens)

                # print(score_dict['prediction'], score_dict['z_score'])
                if detect_result.prediction == True or detect_result.prediction is None:
                    valid_num += 1
                    # write the output text to csv
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
    parser.add_argument("--model_name_or_path", type=str, default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument("--dataset_name", type=str, default="stas/c4-en-10k")
    parser.add_argument("--watermark_name", type=str, choices=["KGW", "SIR"])
    parser.add_argument("--max_valid", type=int, default=2000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--seeding_scheme", type=str, default="selfhash")
    parser.add_argument("--hash_key", type=int, default=2024)
    parser.add_argument("--output_file", type=str, default="wm_text.jsonl")
    parser.add_argument("--output_dir", type=str, default="wm_text")
    return parser.parse_args()


def main():
    args = parse()
    wm = Watermarking(args)
    wm.create_wm_text()


if __name__ == "__main__":
    main()
