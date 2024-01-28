import os
import sys

sys.path.append("/home/lyy/workspace/Watermark/watermarking")
import argparse
import json
import pathlib

import jsonlines
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    LogitsProcessorList,
)


class Rephrase:
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

        file_path = pathlib.Path(self.args.input_dir) / self.args.input_file
        with open(file_path, "r") as f:
            self.json_list = list(f)
        self.settings = json.loads(self.json_list[0])

        self.init_watermark()

    def init_watermark(self):
        if self.args.watermark_name == "watermarking":
            from watermarking.extended_watermark_processor import (
                WatermarkDetector,
                WatermarkLogitsProcessor,
            )

            self.watermark_processor = WatermarkLogitsProcessor(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.settings["gamma"],
                # delta=self.settings['delta'],
                delta=self.args.new_delta,
                seeding_scheme=self.settings["seeding_scheme"],
                hash_key=self.args.hash_key,
            )  # equivalent to `ff-anchored_minhash_prf-4-True-15485863`

            self.watermark_detector_original = WatermarkDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.settings["gamma"],  # should match original setting
                seeding_scheme=self.settings["seeding_scheme"],  # should match original setting
                device=self.model.device,  # must match the original rng device type
                tokenizer=self.tokenizer,
                z_threshold=4.0,
                normalizers=[],
                ignore_repeated_ngrams=True,
                hash_key=self.settings["hash_key"],
            )

            self.watermark_detector_new = WatermarkDetector(
                vocab=list(self.tokenizer.get_vocab().values()),
                gamma=self.settings["gamma"],  # should match original setting
                seeding_scheme=self.settings["seeding_scheme"],  # should match original setting
                device=self.model.device,  # must match the original rng device type
                tokenizer=self.tokenizer,
                z_threshold=4.0,
                normalizers=[],
                ignore_repeated_ngrams=True,
                hash_key=self.args.hash_key,
            )

        elif self.args.watermark_name == "robust_watermark":
            from robust_watermark.watermark import WatermarkWindow

            self.watermark_detector_new = WatermarkWindow(
                device=self.model.device,
                window_size=0,
                gamma=self.settings["gamma"],
                delta=self.args.new_delta,
                target_tokenizer=self.tokenizer,
            )
            self.watermark_processor = WatermarkLogitsProcessor(self.watermark_detector_new)

            self.watermark_detector_original = WatermarkWindow(
                device=self.model.device,
                window_size=0,
                gamma=self.settings["gamma"],
                delta=self.settings["delta"],
                target_tokenizer=self.tokenizer,
            )

    def add_prompt(self, input_data):
        return f"""<<SYS>>\nAssume you are a helpful assistant.\nYou job is to paraphase the given text.\n<</SYS>>\n[INST]\n{input_data}\n[/INST]\nYou're welcome! Here's a paraphrased version of the original message:\n"""

    def rephrase(
        self,
    ):
        """
        Using the LM the continue writing and save the output text.
        """
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        file_path = pathlib.Path(self.args.output_dir) / self.args.output_file

        with jsonlines.open(file_path, mode="w") as writer:
            args_dict = vars(self.args)
            writer.write(args_dict)

            for data in tqdm(self.json_list):
                data = json.loads(data)
                if "z_score" not in data.keys():
                    writer.write(data)
                    continue
                input_text = self.add_prompt(data["generated_text"])
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()

                if self.args.use_wm:
                    output_tokens = self.model.generate(
                        input_ids,
                        temperature=self.settings["temperature"],
                        do_sample=True,
                        top_p=self.settings["top_p"],
                        top_k=self.settings["top_k"],
                        max_new_tokens=self.args.max_new_tokens,
                        min_new_tokens=self.args.min_new_tokens,
                        logits_processor=LogitsProcessorList([self.watermark_processor]),
                    )
                else:
                    output_tokens = self.model.generate(
                        input_ids,
                        temperature=self.settings["temperature"],
                        do_sample=True,
                        top_p=self.settings["top_p"],
                        top_k=self.settings["top_k"],
                        max_new_tokens=self.args.max_new_tokens,
                        min_new_tokens=self.args.min_new_tokens,
                    )

                # if decoder only model, then we need to isolate the
                # newly generated tokens as only those are watermarked, the input/prompt is not
                output_tokens = output_tokens[:, input_ids.shape[-1] :]
                # print(output_tokens)

                output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[
                    0
                ]

                score_dict_ori = self.watermark_detector_original.detect(output_text)
                score_dict_new = self.watermark_detector_new.detect(output_text)
                if type(score_dict_new) != dict:
                    score_dict_new = {"z_score": score_dict_new}
                if type(score_dict_ori) != dict:
                    score_dict_ori = {"z_score": score_dict_ori}
                # print(score_dict['prediction'], score_dict['z_score'])
                # write the output text to csv
                writer.write(
                    {
                        "z_score_ori": round(score_dict_ori["z_score"], 4),
                        "z_score_new": round(score_dict_new["z_score"], 4),
                        "original_text": data["generated_text"],
                        "generated_text": output_text,
                    }
                )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", type=str, default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument("--watermark_name", type=str, default="watermarking")
    parser.add_argument("--new_delta", type=float, default=5.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=16)
    parser.add_argument("--seeding_scheme", type=str, default="selfhash")
    parser.add_argument("--hash_key", type=int, default=2024)
    parser.add_argument("--input_file", type=str, default="wm_text.jsonl")
    parser.add_argument("--input_dir", type=str, default="wm_text")
    parser.add_argument("--output_file", type=str, default="wm_text.jsonl")
    parser.add_argument("--output_dir", type=str, default="wm_text_rephrased")
    parser.add_argument("--use_wm", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse()
    wm = Rephrase(args)
    wm.rephrase()


if __name__ == "__main__":
    main()
