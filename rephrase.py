import argparse
import os
import pathlib

import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import wm_detector as WMD
import wm_generator as WMG


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
        with jsonlines.open(file_path, "r") as reader:
            self.data: list[dict] = list(reader)
            self.settings = self.data[0]

        self.generator: WMG.WMGeneratorBase
        self.detector_origin: WMD.WMDetectorBase
        self.detector_new: WMD.WMDetectorBase

        self.init_watermark()

    def init_watermark(self):
        if self.args.watermark_name == "KGW":
            self.generator = WMG.KGWWMGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                gamma=self.settings["gamma"],
                delta=self.args.new_delta,
                seeding_scheme=self.settings["seeding_scheme"],
                key=self.args.hash_key,
            )
            self.detector_origin = WMD.KGWWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                gamma=self.settings["gamma"],  # should match original setting
                seeding_scheme=self.settings["seeding_scheme"],  # should match original setting
                key=self.settings["hash_key"],
                z_threshold=4.0,
            )
            self.detector_new = WMD.KGWWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                gamma=self.settings["gamma"],  # should match original setting
                seeding_scheme=self.settings["seeding_scheme"],  # should match original setting
                key=self.args.hash_key,
                z_threshold=4.0,
            )

        elif self.args.watermark_name == "SIR":
            self.generator = WMG.SIRWMGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                window_size=0,
                gamma=self.settings["gamma"],
                delta=self.args.new_delta,
                key=self.args.hash_key,
            )
            self.detector_origin = WMD.SIRWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                window_size=0,
                gamma=self.settings["gamma"],  # should match original setting
                delta=self.settings["delta"],  # should match original setting
                key=self.settings["hash_key"],
                z_threshold=4.0,
            )
            self.detector_new = WMD.SIRWMDetector(
                model=self.model,
                tokenizer=self.tokenizer,
                window_size=0,
                gamma=self.settings["gamma"],  # should match original setting
                delta=self.args.new_delta,  # should match args setting
                key=self.args.hash_key,
                z_threshold=4.0,
            )

    def add_prompt(self, input_data):
        return f"""<<SYS>>\nAssume you are a helpful assistant.\nYou job is to paraphase the given text.\n<</SYS>>\n[INST]\n{input_data}\n[/INST]\nYou're welcome! Here's a paraphrased version of the original message:\n"""

    def rephrase(
        self,
    ):
        """
        Using the LM the continue writing and save the output text.
        """
        os.makedirs(self.args.output_dir, exist_ok=True)

        file_path = pathlib.Path(self.args.output_dir) / self.args.output_file

        with jsonlines.open(file_path, mode="w") as writer:
            args_dict = vars(self.args)
            writer.write(args_dict)

            for datum in tqdm(self.data):
                if "z_score" not in datum.keys():
                    writer.write(datum)
                    continue
                input_text = self.add_prompt(datum["generated_text"])
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()

                if self.args.use_wm:
                    output_tokens = self.generator.generate(
                        input_ids,
                        truncate_output=True,
                        temperature=self.settings["temperature"],
                        do_sample=True,
                        top_p=self.settings["top_p"],
                        top_k=self.settings["top_k"],
                        max_new_tokens=self.args.max_new_tokens,
                        min_new_tokens=self.args.min_new_tokens,
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

                result_ori = self.detector_origin.detect_tokens(output_tokens)
                result_new = self.detector_new.detect_tokens(output_tokens)
                # write the output text to csv
                writer.write(
                    {
                        "original_results": result_ori.asdict(),
                        "generated_results": result_new.asdict(),
                        "original_text": datum["generated_text"],
                        "generated_text": self.generator.tokens2text(output_tokens),
                    }
                )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", type=str, default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument("--watermark_name", type=str, choices=["KGW", "SIR", "unbiased"])
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
