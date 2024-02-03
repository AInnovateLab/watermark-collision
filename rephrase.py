import argparse
import hashlib
import logging
from pathlib import Path

import jsonlines
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wm_detector as WMD
import wm_generator as WMG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")


class Rephrase:
    def __init__(self, args):
        # model_name_or_path="TheBloke/Llama-2-7B-GPTQ",
        # dataset_name="stas/c4-en-10k",
        # max_dataset_length=1000
        self.args = args
        self.device = self.args.device
        logging.info(f"Loading model and tokenizer: {self.args.model_name_or_path}")
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

        file_path = self.args.input_file
        logging.info(f"Loading data from: {file_path}")
        with jsonlines.open(file_path, "r") as reader:
            self.settings = reader.read()
            self.data: list[dict] = list(reader)

        logging.info(f"Loading Rephraser Generator config from: {self.args.rephraser_file}")
        logging.info(f"Loading old Detector config from: {self.args.old_detector_file}")
        logging.info(f"Loading new Detector config from: {self.args.new_detector_file}")
        self.rephraser_config = OmegaConf.load(self.args.rephraser_file).generator
        self.old_detector_config = OmegaConf.load(self.args.old_detector_file).detector
        self.new_detector_config = OmegaConf.load(self.args.new_detector_file).detector

        self.rephraser: WMG.WMGeneratorBase
        self.detector_old: WMD.WMDetectorBase
        self.detector_new: WMD.WMDetectorBase

        self.init_watermark()

    def init_watermark(self):
        rephraser_class = WMG.get_generator_class_from_type(self.rephraser_config.type)
        detector_old_class = WMD.get_detector_class_from_type(self.old_detector_config.type)
        detector_new_class = WMD.get_detector_class_from_type(self.new_detector_config.type)
        self.rephraser = rephraser_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.args.new_key,
            **self.rephraser_config,
        )
        self.detector_old = detector_old_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.settings["key"],  # old key
            **self.old_detector_config,
        )
        self.detector_new = detector_new_class(
            model=self.model,
            tokenizer=self.tokenizer,
            key=self.args.new_key,
            **self.new_detector_config,
        )

    def add_prompt(self, input_data):
        return f"""<<SYS>>
Assume you are a helpful assistant.
Your job is to paraphase the given text.
<</SYS>>
[INST]
{input_data}
[/INST]

You're welcome! Here's a paraphrased version of the original message:
"""

    def rephrase(
        self,
    ):
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

        # generate kwargs
        generate_kwargs = {
            "truncate_output": True,
            "temperature": self.args.temperature,
            "do_sample": True,
            "max_new_tokens": self.args.max_new_tokens,
            "min_new_tokens": self.args.min_new_tokens,
        }
        generate_kwargs.update(self.rephraser_config.get("generate_kwargs", {}))

        with jsonlines.open(file_path, mode="w") as writer:
            # writer.write(self.settings)
            writer.write({"args": vars(self.args), "settings": self.settings})

            for datum in tqdm(self.data, dynamic_ncols=True):
                input_text = self.add_prompt(datum["generated_text"])
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()

                if self.args.use_wm:
                    output_tokens = self.rephraser.generate(input_ids, **generate_kwargs)
                else:
                    output_tokens = self.model.generate(input_ids, **generate_kwargs)

                result_ori = self.detector_old.detect_tokens(output_tokens)
                result_new = self.detector_new.detect_tokens(output_tokens)

                writer.write(
                    {
                        "original_results": result_ori.asdict(),
                        "generated_results": result_new.asdict(),
                        "original_text": datum["generated_text"],
                        "generated_text": self.rephraser.tokens2text(output_tokens),
                    }
                )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-name-or-path", type=str, default="TheBloke/Llama-2-7B-GPTQ")
    # Generator/Detector loading
    parser.add_argument(
        "--rephraser-file", type=str, required=True, help="Yaml file for rephraser."
    )
    parser.add_argument(
        "--old-detector-file", type=str, required=True, help="Yaml file for old detector."
    )
    parser.add_argument(
        "--new-detector-file", type=str, required=True, help="Yaml file for new detector."
    )
    # generate kwargs
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    # Watermark kwargs
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--new-key", type=int, default=2024)
    # I/O
    parser.add_argument("--input-file", type=str, required=True, help="Path to input file.")
    parser.add_argument("--use-wm", action="store_true", default=False)
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        default=False,
        help="Overwrite output file without confirmation if set true",
    )
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
    wm = Rephrase(args)
    wm.rephrase()


if __name__ == "__main__":
    main()
