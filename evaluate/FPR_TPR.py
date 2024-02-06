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
    parser.add_argument(
        "--gen-wm-jsonl", type=str, required=True, help="Generated watermark .jsonl file."
    )
    parser.add_argument(
        "--gen-no-wm-jsonl", type=str, required=True, help="Generated no-watermark .jsonl file."
    )
    parser.add_argument(
        "--wm-jsonl", type=str, required=True, help="Rephrased watermark .jsonl file."
    )
    parser.add_argument(
        "--no-wm-jsonl", type=str, required=True, help="Rephrased no-watermark .jsonl file."
    )

    return parser.parse_args()


def evaluate_fpr_tpr(
    fprs: list[float],
    gen_wm_data: list[dict],
    gen_no_wm_data: list[dict],
    wm_data: list[dict],
    no_wm_data: list[dict],
) -> list[dict[str, float]]:
    """Calculate the TPRs, given specific FPRs.

    Note:
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
    """
    results: list[dict[str, float]] = []

    def get_score(detect_result: dict) -> float:
        if "z_score" in detect_result:
            return detect_result["z_score"]
        elif "llr_score" in detect_result:
            return detect_result["llr_score"]
        else:
            raise ValueError("Unknown score type.")

    def np_dropna(arr: np.ndarray) -> np.ndarray:
        return arr[~np.isnan(arr) & ~np.isinf(arr)]

    gen_wm = np.array([get_score(d["results"]) for d in gen_wm_data], dtype=np.float64)
    gen_wm = np_dropna(gen_wm)
    gen_no_wm = np.array([get_score(d["results"]) for d in gen_no_wm_data], dtype=np.float64)
    gen_no_wm = np_dropna(gen_no_wm)
    wm_olds = np.array([get_score(d["original_results"]) for d in wm_data], dtype=np.float64)
    wm_olds = np_dropna(wm_olds)
    no_wm_olds = np.array([get_score(d["original_results"]) for d in no_wm_data], dtype=np.float64)
    no_wm_olds = np_dropna(no_wm_olds)
    wm_news = np.array([get_score(d["generated_results"]) for d in wm_data], dtype=np.float64)
    wm_news = np_dropna(wm_news)
    no_wm_news = np.array([get_score(d["generated_results"]) for d in no_wm_data], dtype=np.float64)
    no_wm_news = np_dropna(no_wm_news)
    for fpr in fprs:
        # generated
        thres_gen = np.percentile(gen_no_wm, 100 * (1 - fpr))
        tpr_gen = np.sum(gen_wm > thres_gen) / len(gen_wm)
        # generated & rephrased by no_wm
        thres_gen_wo_wm = np.percentile(gen_no_wm, 100 * (1 - fpr))
        tpr_gen_no_wm_rephrased = np.sum(no_wm_olds > thres_gen_wo_wm) / len(gen_wm)
        # old
        thres_old = thres_gen
        tpr_old = np.sum(wm_olds > thres_old) / len(wm_olds)
        # new
        thres_new = np.percentile(no_wm_news, 100 * (1 - fpr))
        tpr_new = np.sum(wm_news > thres_new) / len(wm_news)

        results.append(
            {
                "generated": tpr_gen,
                "no_wm_rephrased": tpr_gen_no_wm_rephrased,
                "old": tpr_old,
                "new": tpr_new,
            }
        )
    return results


def main(args, verbose=False) -> dict[float, dict[str, float]]:
    gen_wm_jsonl, gen_no_wm_jsonl = args.gen_wm_jsonl, args.gen_no_wm_jsonl
    wm_jsonl, no_wm_jsonl = args.wm_jsonl, args.no_wm_jsonl
    with (
        jsonlines.open(gen_wm_jsonl) as gen_wm_reader,
        jsonlines.open(gen_no_wm_jsonl) as gen_no_wm_reader,
        jsonlines.open(wm_jsonl) as wm_reader,
        jsonlines.open(no_wm_jsonl) as no_wm_reader,
    ):
        # generated
        gen_wm_config, gen_no_wm_config = gen_wm_reader.read(), gen_no_wm_reader.read()
        gen_wm_data, gen_no_wm_data = list(gen_wm_reader), list(gen_no_wm_reader)

        # rephrased
        wm_config, no_wm_config = wm_reader.read(), no_wm_reader.read()
        if gen_wm_config["key"] != wm_config["settings"]["key"]:
            logging.warning("The `key` of the generating and rephrasing configs are different.")
        if wm_config["args"]["use_wm"] != True:
            logging.warning("The `use_wm` of the `wm-jsonl` config is not True.")
        if no_wm_config["args"]["use_wm"] != False:
            logging.warning("The `use_wm` of the `no-wm-jsonl` config is not False.")
        if wm_config["args"]["new_key"] != no_wm_config["args"]["new_key"]:
            logging.warning("The `new_key` of the rephrasing configs are different.")
        if wm_config["settings"]["key"] != no_wm_config["settings"]["key"]:
            logging.warning("The `key` of the rephrasing configs are different.")
        wm_data, no_wm_data = list(wm_reader), list(no_wm_reader)

        fprs: list[float] = args.fprs
        results = evaluate_fpr_tpr(args.fprs, gen_wm_data, gen_no_wm_data, wm_data, no_wm_data)
        for fpr, result in zip(fprs, results):
            gen_tpr, nwr_tpr, old_tpr, new_tpr = (
                result["generated"],
                result["no_wm_rephrased"],
                result["old"],
                result["new"],
            )
            if verbose:
                print(
                    f"FPR: {fpr*100: >6.2f}%,    Gen TPR: {gen_tpr*100: >6.2f}%,    "
                    f"NW Rephrased TPR: {nwr_tpr*100: >6.2f}%,    "
                    f"Old TPR: {old_tpr*100: >6.2f}%,    New TPR: {new_tpr*100: >6.2f}%"
                )
        return {fpr: result for fpr, result in zip(fprs, results)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    args = parse_args()
    main(args, verbose=True)
