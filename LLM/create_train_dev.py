import argparse
import torch
import os
import tqdm
from pathlib import Path
from plda_utils import max_and_avg_pool

import logging

from chateval.datasets import load_dataset as load_chateval_dataset
from chateval.utils import set_seed
from chateval.results import (
    evaluate_metric,
    results_to_csv,
    results_to_wandb,
    get_wb_run_logger,
    exp_dirname,
)
import json

from model import get_model
from prompt import get_prompt
from prompt.vh_single_metric import *  # necessary to run the registration - also useful to list where the templates are coming from
from prompt.op_multi_metrics import *  # necessary to run the registration - also useful to list where the templates are coming from


def get_examples(df, context_list, response_list, model, gold_qualities):
    examples = []
    for turn_idx, (context, response) in enumerate(tqdm.tqdm(zip(context_list, response_list))):
        predicted_qualities, errors, logs, hidden_states = model(
            dialogue_context=context, response=response
        )
        d = {
                **predicted_qualities,
                # "embeddings_list": hidden_states,
                "embedding": max_and_avg_pool(hidden_states),
                "dialogue_id": df["dialogue_id"][turn_idx],  # funny name for turn_id, right? copying the challenge naming
            }
        for gq, v in gold_qualities.items():
            d[gq] = v[turn_idx]
        examples.append(d)
    logging.info(f"Prepared {len(examples)} examples.")
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/home/oplatek/.cache/huggingface/",
        help="You could try one of /home/oplatek/.cache/huggingface/ /home/hudecek/hudecek/hf_cache",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="allenai/tk-instruct-11b-def-pos-neg-expl",
        help="Model name or path to be used. Accepts subset of HF and OpenAI models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="vh-appropriateness-03",
        help="Names of the prompts templates. See templates in the ./prompt/ directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--display_first_examples",
        type=int,
        default=2,
        help="Number of examples to use for few shot prompt construction.",
    )
    parser.add_argument(
        "--dev_dataset", type=str, default='rehearsal-copy-hard', help="Name of dataset to evaluate"
    )
    parser.add_argument(
        "--train_datasets", type=str, required=True, help="Name of dataset to train on"
    )
    parser.add_argument("--eval_type", type=str, default="en", help="en or par")
    parser.add_argument("--eval_level", type=str, default="turn")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    # loading dataset
    set_seed(args.seed)
    if args.outdir is None:
        args.outdir = exp_dirname(args, main_file=__file__)
    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.outdir}/run.log"),
            logging.StreamHandler(),
        ],
    )

    # wandb is useful to have early to monitor use of resources - it assumes stable and working scripts
    # in case of debugging use: wandb offline, or wandb disable so you do not polute our project logs at the cloud
    # loading dataset
    dev_df, dev_context_list, dev_response_list, dev_gold_qualities = load_chateval_dataset(
        args.dev_dataset, args.eval_type
    )
    logging.info(f"Creating embeddings from {args.dev_dataset}.")
    prompt = get_prompt(args.prompt)
    model = get_model(prompt, args.model_name, args.cache_dir)
    logging.info(f"Example of the used prompt {model.prompt.__post_init__()}")

    dev_examples = get_examples(dev_df, dev_context_list, dev_response_list, model, dev_gold_qualities)
    train_examples = []
    for dataset in args.train_datasets.split(','):
        logging.info(f"Creating embeddings from {dataset}.")
        df, context_list, response_list, gold_qualities = load_chateval_dataset(
            dataset, args.eval_type, banned_ctxs=[ctx for ctx in dev_df["context"]]
        )
        train_examples.extend(get_examples(df, context_list, response_list, model, gold_qualities))


    outdir = Path(args.outdir)
    os.symlink(outdir.relative_to(outdir.parent), outdir.parent / f"data_{args.dev_dataset}_{args.train_datasets}")
    outpath = outdir / "dev_examples.pth"
    print(f"Dev set path {str(outpath)}", flush=True)
    torch.save(dev_examples, outpath)
    outpath = outdir / "train_examples.pth"
    print(f"Train set path {str(outpath)}", flush=True)
    torch.save(train_examples, outpath)

