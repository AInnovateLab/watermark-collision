import os
import sys
sys.path.append("/home/lyy/workspace/Watermark/watermarking")
# import pandas as pd
import jsonlines
import pathlib

from tqdm import tqdm
from torch.utils.data import DataLoader
from watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, DataCollatorWithPadding
from datasets import load_dataset


class Watermarking:
    def __init__(self, args):
        # model_name_or_path="TheBloke/Llama-2-7B-GPTQ",
        # dataset_name="stas/c4-en-10k",
        # max_dataset_length=1000
        self.args = args
        self.device = self.args.device
        print(
            f"Loading model and tokenizer:{self.args.model_name_or_path}...", end='')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, use_fast=True, padding_side="left")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # To use a different branch, change revision
        # For example: revision="main"
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path,
                                                          device_map=0,
                                                          trust_remote_code=True,
                                                          revision="main")
        print("Done!")

        print(f"Loading dataset:{self.args.dataset_name}...", end='')
        self.dataset = load_dataset(self.args.dataset_name)
        print("Done!")
        self.dataset = self.dataset['train']

        # def add_prompt(input_data):
        #     input_data[
        #         'text'] = f'''<<SYS>>\nAssume you are a helpful assistant.\nYou job is to paraphase the given text.\n<</SYS>>\n[INST]\n{input_data['text']}\n[/INST]\nYou're welcome! Here's a paraphrased version of the original message:\n'''
        #     return input_data
        # self.dataset = self.dataset.map(add_prompt)

        def tokenize(batch):
            return self.tokenizer(batch['text'], return_tensors='pt', truncation=True, max_length=128)
        self.tokenized_dataset = self.dataset.map(tokenize)
        self.tokenized_dataset.set_format("torch")

        self.watermark_processor = WatermarkLogitsProcessor(vocab=list(self.tokenizer.get_vocab().values()),
                                                            gamma=self.args.gamma,
                                                            delta=self.args.delta,
                                                            seeding_scheme=self.args.seeding_scheme,
                                                            hash_key=self.args.hash_key)  # equivalent to `ff-anchored_minhash_prf-4-True-15485863`

        self.watermark_detector = WatermarkDetector(vocab=list(self.tokenizer.get_vocab().values()),
                                                    gamma=self.args.gamma,  # should match original setting
                                                    seeding_scheme=self.args.seeding_scheme,  # should match original setting
                                                    device=self.model.device,  # must match the original rng device type
                                                    tokenizer=self.tokenizer,
                                                    z_threshold=4.0,
                                                    normalizers=[],
                                                    ignore_repeated_ngrams=True,
                                                    hash_key=self.args.hash_key)

    def create_wm_text(self):
        '''
        Using the LM the continue writing and save the output text.
        '''
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        file_path = pathlib.Path(self.args.output_dir) / self.args.output_file

        with jsonlines.open(file_path, mode='w') as writer:
            args_dict = vars(self.args)
            writer.write(args_dict)
            valid_num = 0

            for data in tqdm(self.tokenized_dataset):
                input_ids = data['input_ids'].to(self.device)
                # print(input_ids)
                output_tokens = self.model.generate(input_ids,
                                    temperature=self.args.temperature,
                                    do_sample=True,
                                    top_p=self.args.top_p,
                                    top_k=self.args.top_k,
                                    max_new_tokens=self.args.max_new_tokens,
                                    min_new_tokens=self.args.min_new_tokens,
                                    logits_processor=LogitsProcessorList([self.watermark_processor]))

                # if decoder only model, then we need to isolate the
                # newly generated tokens as only those are watermarked, the input/prompt is not
                output_tokens = output_tokens[:, input_ids.shape[-1]:]
                # print(output_tokens)

                output_text = self.tokenizer.batch_decode(
                    output_tokens, skip_special_tokens=True)[0]

                score_dict = self.watermark_detector.detect(output_text)
                # print(score_dict['prediction'], score_dict['z_score'])
                if score_dict['prediction']:
                    valid_num += 1
                    # write the output text to csv
                    writer.write({'z_score': round(score_dict['z_score'], 4), 'original_text':data['text'], 'generated_text': output_text})
                
                if valid_num >= self.args.max_valid:
                    break


