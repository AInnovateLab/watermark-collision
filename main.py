import sys
sys.path.append("/home/lyy/workspace/Watermark/watermarking")
from watermarking.extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

def main(input_text):
    # Load model directly
    model_name_or_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
    # To use a different branch, change revision
    # For example: revision="main"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                device_map="auto",
                                                trust_remote_code=True,
                                                revision="main")
    
    # prompt = "Assume you are a helpful assistant. \
    #           You job is to paraphase the given text. \
    #           Here is the given text and please rephase it: "
    # prompt_template=f'''{prompt}{input_text}\n\n Answer: '''
    prompt_template = \
f'''<<SYS>>
Assume you are a helpful assistant.
You job is to paraphase the given text.
<</SYS>>

[INST]
{input_text}
[/INST]

You're welcome! Here's a paraphrased version of the original message:

'''

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="selfhash",
                                                hash_key=2024) #equivalent to `ff-anchored_minhash_prf-4-True-15485863`
    # Note:
    # You can turn off self-hashing by setting the seeding scheme to `minhash`.

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    # note that if the model is on cuda, then the input is on cuda
    # and thus the watermarking rng is cuda-based.
    # This is a different generator than the cpu-based rng in pytorch!

    output_tokens = model.generate(input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=128,
                                logits_processor=LogitsProcessorList([watermark_processor]))

    # if decoder only model, then we need to isolate the
    # newly generated tokens as only those are watermarked, the input/prompt is not
    output_tokens = output_tokens[:, input_ids.shape[-1]:]

    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    print(output_text)

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=0.25, # should match original setting
                                            seeding_scheme="selfhash", # should match original setting
                                            device=model.device, # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=4.0,
                                            normalizers=[],
                                            ignore_repeated_ngrams=True,
                                            hash_key=2024)

    score_dict = watermark_detector.detect(output_text) # or any other text of interest to analyze
    print(score_dict)

if __name__ == "__main__":
    main("Thank you for providing the details for the upcoming technical interview at Huawei International Pte Ltd.")
    # main("Welcome back to campus! We hope you've recharged over the break and are ready to dive right into campus life, starting with our largest recruitment fair of the 2024")
