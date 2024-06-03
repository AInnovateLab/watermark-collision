from typing import Dict, Any, Optional, Tuple, Union

import argparse
import jsonlines
from transformers.pipelines.base import GenericTensor
from transformers import PreTrainedTokenizerBase, FillMaskPipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, LogitsProcessorList
from datasets import Dataset
from omegaconf import OmegaConf, DictConfig
from copy import deepcopy
import wm_detector as WMD
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import torch
import logging
import numpy as np

from mlm_wm_generator import get_wm_logits_processor


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")



class MLMGenerator:
    """
    Generate masked tokens for MLM watermark attack.
    Modified from transformers.data.data_collator. We don't need to replace words in this task.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 20% original.
        """
        import torch

        labels = inputs.clone()
        inputs = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels
    

class FillMaskTkPipeline(FillMaskPipeline):
    """
    Accept token ids instead of strings. Others are the same as FillMaskPipeline.
    """
    logits_processor: LogitsProcessorList

    
    def __init__(self, wm_processor: LogitsProcessorList = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = wm_processor
    
    
    def preprocess(
        self, inputs, return_tensors=None, tokenizer_kwargs=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        self.ensure_exactly_one_mask_token(inputs)
        return inputs
    
    
    def postprocess(self, model_outputs, top_k=5, target_ids=None):
        # Cap top_k if there are targets
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]

        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        # Fill mask pipeline supports only one ${mask_token} per sample

        logits = outputs[0, masked_index, :]
        # Calculate new score
        if self.logits_processor is not None:
            for i, idx in enumerate(masked_index):
                logits[i] = self.logits_processor(input_ids[:idx].unsqueeze(0), logits[i].unsqueeze(0)).squeeze(0)
        probs = logits.softmax(dim=-1)
        if target_ids is not None:
            probs = probs[..., target_ids]

        values, predictions = probs.topk(top_k)

        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="TheBloke/Llama-2-13B-chat-GPTQ", help="Model name or path")
    parser.add_argument("--mlm-model", type=str, default="FacebookAI/roberta-large", help="MLM model name or path")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy for downloading models and datasets")
    # dataset
    parser.add_argument("--input-file", type=str, help="Path to input file.", required=True)
    # MLM attack settings
    parser.add_argument("--mlm-probability", type=float, default=0.15, help="Probability of masking tokens.")
    # Watermark detector settings[Optional]
    parser.add_argument("--detector-file", type=str, default='', help="Yaml file for wm detector.")
    parser.add_argument("--detector-key", type=int, default=2023)
    # Logits Processor
    parser.add_argument("--wm-processor", type=str, default='', help="Watermark processor config file. None for no watermark.")
    parser.add_argument("--wm-key", type=int, default=2024, help="Watermark key for watermark processor.")
    # I/O
    parser.add_argument("--no-confirm", action="store_true", default=False, help="Overwrite output file without confirmation if set true")
    output_ex_group = parser.add_mutually_exclusive_group(required=True)
    output_ex_group.add_argument("--output-dir", type=str, help="Output directory. If specified, enable automatic naming from the yaml file of generator and detector.",)
    output_ex_group.add_argument("--output-file", type=str, help="Output file name. If specified, disable the automatic naming and ignore the --output-dir setting.")

    return parser.parse_args()



class FillMaskTask:
    def __init__(self, args) -> None:
        """
        Basic configs
        """
        self.args = args
        self.device = args.device
        if self.args.proxy:
            self.proxy = {
                "http": self.args.proxy,
                "https": self.args.proxy,
            }
        else:
            self.proxy = None
            
        """
        Model and tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(args.mlm_model, use_fast=True, proxies=self.proxy)
        self.model = AutoModelForMaskedLM.from_pretrained(args.mlm_model, proxies=self.proxy)
        self.mlm_generator = MLMGenerator(tokenizer=self.tokenizer, mlm_probability=args.mlm_probability)
        
        """
        Logits Processor
        """
        if args.wm_processor:
            wm_processor_config = OmegaConf.load(args.wm_processor)
            wm_processor_config = wm_processor_config.generator
            wm_processor = get_wm_logits_processor(tokenizer = self.tokenizer, key = args.wm_key, device=self.model.device, **wm_processor_config)
        else:
            wm_processor = None
        
        self.pipeline = FillMaskTkPipeline(wm_processor=wm_processor, model=self.model, tokenizer=self.tokenizer, device=self.device)
        
        """
        Dataset
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
        
        
        """
        WM Detector
        """
        self.detector_config: DictConfig
        self.detector: WMD.WMDetectorBase
        self.detector_key: Union[str, int, None]
        
        self.orig_config = {
                "rephraser": [],
                "key": [],
                "detector": [],
                "detector_key": [],
            }
        
        # If we can load old detector from the input file, add to the list
        if args.detector_file:
            old_detector_config = OmegaConf.load(args.detector_file)
            self.detector_config = old_detector_config.detector
            self.detector_key = args.detector_key
        elif self.input_config.get('detector_file', None):
            # Old format
            detector_file = self.input_config['detector_file']
            if not osp.exists(detector_file):
                logging.error("Detector file not found, please make sure the path in the config file exists or specify a new one.")
                exit(0)
            else:
                old_detector_config = OmegaConf.load(detector_file)
                self.detector_config = old_detector_config.detector
                self.detector_key = self.input_config.get('key', None)
        elif self.input_config.get('detector', None):
            # New format
            self.orig_config = deepcopy(self.input_config)
            last_rep_idx = self._get_last_detector_idx()
            self.detector_config = OmegaConf.create(self.orig_config['rephraser'][last_rep_idx]).detector
            self.detector_key = self.orig_config['key'][last_rep_idx]
        else:
            logging.error("No old detector config found, please specify one or check input file format")
            exit(0)
            
        detector_class = WMD.get_detector_class_from_type(self.detector_config.type)
        orig_model = AutoModelForCausalLM.from_pretrained(self.args.model, device_map='auto', trust_remote_code=True, revision="main", proxies=self.proxy)
        orig_tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True, padding_side="left", proxies=self.proxy)
        self.detector = detector_class(
            model=orig_model,
            tokenizer=orig_tokenizer,
            key=self.detector_key,
            **self.detector_config,
        )
        self.orig_tokenizer = orig_tokenizer
        
        """
        Output
        """
        if self.args.output_file:
            file_path = Path(self.args.output_file)
        elif self.args.output_dir:
            file_path = Path(self.args.output_dir)
            file_path.mkdir(parents=True, exist_ok=True)
            # automatic naming
            input_name = Path(self.args.input_file).stem
            file_path = (
                file_path
                / f"{input_name}#MLM_{self.args.mlm_probability}.jsonl"
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
        
    def mlm(self):
        with (
            jsonlines.open(self.file_path, mode="w", flush='true') as writer,
            tqdm(total=len(self.dataset), desc="Valid samples", dynamic_ncols=True) as pbar,
        ):
            orig_config = deepcopy(self.orig_config)
            orig_config['detector'].append([OmegaConf.to_container(self.detector_config)])
            orig_config['detector_key'].append([self.detector_key])
            orig_config['rephraser'].append("MLM")
            orig_config['key'].append(self.args.mlm_probability)
            writer.write(orig_config)
            
            for datum in self.dataset:
                # MLM Mask
                encoded = self.tokenizer(datum["text"], return_tensors="pt")
                encoded_text = encoded['input_ids'].cuda()
                attention_mask = encoded['attention_mask'].cuda()
                while True:
                    input_ids, labels = self.mlm_generator.torch_mask_tokens(encoded_text)
                    # At least one mask token
                    masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
                    numel = np.prod(masked_index.shape)
                    if numel >= 1:
                        break
                # Generate
                preds = self.pipeline({"input_ids": input_ids, "attention_mask": attention_mask}, top_k=1)
                # Fill the mask
                for pred, index in zip(preds, torch.nonzero(labels >= 0)):
                    if isinstance(pred, list):
                        pred = pred[0]
                    encoded_text[index[0], index[1]] = pred['token']
                # decode text
                output_text = self.tokenizer.decode(encoded_text[0], skip_special_tokens=True)
                # Detect
                detect_result = self.detector.detect_text(output_text)
                detect_results = [detect_result.asdict()]
                
                """
                Save results and generated text to the output file, along with history
                """
                det_result = deepcopy(datum['results'])
                texts = deepcopy(datum['texts'])
                det_result.append(detect_results)
                texts.append({'mlm': output_text})
                writer.write(
                    {
                        "results": det_result,
                        "texts": texts,
                    }
                )
                pbar.update()
        

def main():
    args = parse()
    wm = FillMaskTask(args)
    wm.mlm()


if __name__ == "__main__":
    main()