import argparse
import logging
import warnings

import jsonlines
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the TPR of a model under a specific FPR."
    )
    parser.add_argument(
        "--fprs",
        type=float,
        default=[0.01, 0.05, 0.1],
        nargs="+",
        help="False positive rates to be computed.",
    )
    parser.add_argument("--wm-jsonl", type=str, required=True, help="Watermark .jsonl file.")
    parser.add_argument("--no-wm-jsonl", type=str, required=True, help="No watermark .jsonl file.")

    return parser.parse_args()


def evaluate_fpr_tpr(
    fprs: list[float], wm_data: list[dict], no_wm_data: list[dict]
) -> list[tuple[float, float]]:
    """Calculate the TPRs, given specific FPRs.

    Note:
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
    """
    results: list[tuple[float, float]] = []

    def get_score(detect_result: dict) -> float:
        if "z_score" in detect_result:
            return detect_result["z_score"]
        elif "llr_score" in detect_result:
            return detect_result["llr_score"]
        else:
            raise ValueError("Unknown score type.")

    wm_olds = np.array([get_score(d["original_results"]) for d in wm_data], dtype=np.float64)
    wm_news = np.array([get_score(d["generated_results"]) for d in wm_data], dtype=np.float64)
    no_wm_olds = np.array([get_score(d["original_results"]) for d in no_wm_data], dtype=np.float64)
    no_wm_news = np.array([get_score(d["generated_results"]) for d in no_wm_data], dtype=np.float64)
    for fpr in fprs:
        # old
        thres_old = np.percentile(no_wm_olds, 100 * (1 - fpr))
        tpr_old = np.sum(wm_olds > thres_old) / len(wm_olds)
        # new
        thres_new = np.percentile(no_wm_news, 100 * (1 - fpr))
        tpr_new = np.sum(wm_news > thres_new) / len(wm_news)

        results.append((tpr_old, tpr_new))
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    args = parse_args()
    wm_jsonl, no_wm_jsonl = args.wm_jsonl, args.no_wm_jsonl
    with jsonlines.open(wm_jsonl) as wm_reader, jsonlines.open(no_wm_jsonl) as no_wm_reader:
        wm_config, no_wm_config = wm_reader.read(), no_wm_reader.read()
        # config check
        if wm_config["args"]["use_wm"] != True:
            logging.warning("The `use_wm` of the `wm-jsonl` config is not True.")
        if no_wm_config["args"]["use_wm"] != False:
            logging.warning("The `use_wm` of the `no-wm-jsonl` config is not False.")
        if wm_config["args"]["new_key"] != no_wm_config["args"]["new_key"]:
            logging.warning("The `new_key` of the two configs are different.")
        if wm_config["settings"]["key"] != no_wm_config["settings"]["key"]:
            logging.warning("The `key` of the two configs are different.")
        wm_data, no_wm_data = list(wm_reader), list(no_wm_reader)

        fprs: list[float] = args.fprs
        results = evaluate_fpr_tpr(args.fprs, wm_data, no_wm_data)
        for fpr, result in zip(fprs, results):
            old_tpr, new_tpr = result
            print(f"FPR: {fpr*100:.2f}%, Old TPR: {old_tpr*100:.2f}%, New TPR: {new_tpr*100:.2f}%")


if __name__ == "__main__":
    main()
