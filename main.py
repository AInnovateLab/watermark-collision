# import sys
# sys.path.append("/home/lyy/workspace/Watermark/watermarking")
import argparse
from create_wm_text import Watermarking

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--model_name_or_path', type=str, default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument('--dataset_name', type=str, default="stas/c4-en-10k")
    parser.add_argument('--max_valid', type=int, default=2000)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--min_new_tokens', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--seeding_scheme', type=str, default="selfhash")
    parser.add_argument('--hash_key', type=int, default=2024)
    parser.add_argument('--output_file', type=str, default='wm_text.jsonl')
    parser.add_argument('--output_dir', type=str, default='wm_text')
    return parser.parse_args()


def main():
    args = parse()
    wm = Watermarking(args)
    # print(wm.tokenized_dataset)
    # for batch in wm.dataloader:
    #     break
    # print({k: v.shape for k, v in batch.items()})
    # print()
    wm.create_wm_text()


if __name__ == "__main__":
    main()