import json
from functools import partial

import torch.multiprocessing as mp
from tqdm import tqdm
from tree_sitter import Language, Parser

from eval_utils import (
    postprocess_code_lines,
    extract_identifiers,
    cal_edit_sim,
    remove_comments
)

parser = None


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


def compute_edit_sim(samples):
    refs, hyps = [], []
    for s in samples:
        refs.append(s["target"])
        hyps.append(s["pred"])
    return cal_edit_sim(refs, hyps)


def process_examples(lang, args):
    sample, ex = args
    global parser

    prediction = postprocess_code_lines(ex["prompt"], sample["pred"], parser, lang)
    prediction = remove_comments(prediction)
    target = ex["groundtruth"]
    target = remove_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)

    pred_ids = extract_identifiers(prediction, lang)
    target_ids = extract_identifiers(target, lang)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target,
        "pred_ids": pred_ids,
        "target_ids": target_ids
    }
    return trunc_s, em_label


def compute_metric_stmt(args):
    with open(f"{args.output_dir}/prediction.jsonl", "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            samples.append(json.loads(l))

    examples = {}
    with open(args.prompt_file, "r") as f_in:
        for l in f_in.readlines():
            ex = json.loads(l)
            examples[ex["metadata"]["task_id"]] = {
                "prompt": ex["prompt"],
                "groundtruth": ex["groundtruth"]
            }

    assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"

    global parser
    ts_lang = "c_sharp" if args.language == "csharp" else args.language
    language = Language(args.ts_lib, ts_lang)
    parser = Parser()
    parser.set_language(language)

    truncated_samples = []
    em_labels = []

    print("post-processing samples ...")
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples, args.language)

    with tqdm(total=len(samples)) as pbar:
        for output in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            trunc_s, em_label = output
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    exact_match = 0
    with open(f"{args.output_dir}/prediction_truncated.jsonl", 'w', encoding="utf-8") as pt, \
            open(f"{args.output_dir}/exact_match_idx.jsonl", 'w') as em:
        for trunc_s, em_label in zip(truncated_samples, em_labels):
            pt.write(json.dumps(trunc_s) + "\n")
            if em_label == 1:
                em.write(f'{trunc_s["task_id"]}\n')
                exact_match += 1

    ### Score calculation

    id_em = []
    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        identifier_em = int(trunc_s["pred_ids"] == trunc_s["target_ids"])
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        id_tp, id_fp, id_fn = compute_id_match(trunc_s["pred_ids"], trunc_s["target_ids"])
        id_em.append(identifier_em)
        edit_similarities.append(es)

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em_labels[idx],
            "es": es,
            "id_em": identifier_em,
            "id_precision": id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0,
            "id_recall": id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0,
            "id_f1": 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0,
        })

    em_ratio = round(exact_match / len(samples) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)

    id_em_ratio = round(
        sum(detailed_results[idx]['id_em'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)
    id_precision = round(sum(detailed_results[idx]['id_precision'] for idx in range(len(detailed_results))) / len(
        detailed_results) * 100, 2)
    id_recall = round(
        sum(detailed_results[idx]['id_recall'] for idx in range(len(detailed_results))) / len(detailed_results) * 100,
        2)
    id_f1 = round(
        sum(detailed_results[idx]['id_f1'] for idx in range(len(detailed_results))) / len(detailed_results) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}"
    )

    print(
        f"ID matching: "
        f"EM {id_em_ratio}, "
        f"Precision {id_precision}, "
        f"Recall {id_recall}, "
        f"F1 {id_f1}"
    )

    with open(f"{args.output_dir}/detailed_results.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # write the results to a file
    with open(f"{args.output_dir}/results.json", 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "id_em": id_em_ratio,
            "id_precision": id_precision,
            "id_recall": id_recall,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))
