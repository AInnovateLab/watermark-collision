import sys
import torch
from watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
sys.path.append("/home/lyy/workspace/Watermark/watermarking")

def main(input_text):
    # Load model directly
    model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=True,
                                                revision="main")
    
    prompt = "Tell me about AI"
    prompt_template=f'''{prompt}'''

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    output_text = "This is the output text and we are testing whether the watermark can be detected."
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25, # should match original setting
                                            seeding_scheme="selfhash", # should match original setting
                                            device=model.device, # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_ngrams=True)

    score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
    print(score_dict)

if __name__ == "__main__":
    main("Tell me about AI")