import argparse
import torch
import os
import tqdm
from pathlib import Path
from collections import defaultdict
from plda_utils import max_and_avg_pool

import logging

from chateval.datasets import load_dataset, load_task2_test_dataset
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


def predict(
    dataset, model, prompt, df, context_list, response_list, args, gold_qualities=None
):
    predicted_scores = {metric: [] for metric in prompt.qualities}
    total_errors = {metric: 0 for metric in prompt.qualities}
    embeddings = []
    if "dialogue_id" in df:
        # def data preprocessed as in baseline
        # funny name, right? copying the challenge naming
        get_turn_id = lambda turn_idx: df["dialogue_id"][turn_idx]
    elif "UID" in df:
        # test data
        get_turn_id = lambda turn_idx: df["UID"].tolist()[turn_idx]
    else:
        raise ValueError(f"Unknown column for turn id format {df.columns=}")

    for context, response in tqdm.tqdm(zip(context_list, response_list)):
        turns = len(predicted_scores[prompt.qualities[0]])
        predicted_qualities, errors, logs, hidden_states = model(
            dialogue_context=context, response=response
        )
        if turns < args.display_first_examples:
            print(
                f"\n\n[{dataset} {turns+1}]\nPrompt:\n\t'''{logs['prompt']}'''\nContext:\n\t'''{logs['dialogue_context']}'''\nResponse:\n\t'''{logs['response']}'''\nRaw reply:\n\t'''{logs['raw_reply']}'''",
                flush=True,
            )
            print(
                f"Predicted qualities:\n\t{predicted_qualities}\nErrors:\n\t{errors}\n\n",
                flush=True,
            )
        for quality in prompt.qualities:
            predicted_scores[quality].append(predicted_qualities[quality])
            total_errors[quality] += errors[quality]

        if args.save_embeddings:
            d = {
                **predicted_qualities,
                # "embeddings_list": hidden_states,
                "embedding": max_and_avg_pool(hidden_states),
                "dialogue_id": get_turn_id(turns),
            }
            if gold_qualities is not None:
                for gq, v in gold_qualities.items():
                    d[gq] = v[turns]
            embeddings.append(d)

    return predicted_scores, total_errors, embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_embeddings", default=False, action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--use_4bit", action="store_false", default=True, help="Activate 4bit precision base model loading")
    parser.add_argument("--use_nested_quant", action="store_true", help="Activate nested quantization for 4bit base models")
    parser.add_argument("--bnb_4bit_compute_dtype", default="float16", help="Compute dtype for 4bit base models")
    parser.add_argument("--bnb_4bit_quant_type", default="nf4", help="Quantization type fp4 or nf4")
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
        "--max_test_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--display_first_examples",
        type=int,
        default=100,
        help="Number of examples to use for few shot prompt construction.",
    )
    parser.add_argument(
        "--dev_set", type=str, default=None, help="Name of dataset to evaluate"
    )
    parser.add_argument("--eval_type", type=str, default="en", help="en or par")
    parser.add_argument(
        "--eval_level", type=str, default="turn", help="Dialogue is not supported"
    )
    parser.add_argument("--insert_speaker", action="store_true")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--test_task1", default=False, action="store_true")
    parser.add_argument("--test_task2", default=False, action="store_true")
    parser.add_argument("--train_sets", default="", type=str)
    args = parser.parse_args()
    if len(args.train_sets) > 0 and not args.save_embeddings:
        raise ValueError(
            "YOU PROBABLY WANTED TO GENERATE EMBEDDING FOR TRAINING, right? If no comment this line out(but do not commit it"
        )
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
    # Each experiment in wandb can be found at https://wandb.ai/metric/chateval/runs/WANDB_RUN_ID
    # So only copy the WANDB_RUN_ID and see exp/WANDB_RUN_ID we create symlink here
    run_logger = get_wb_run_logger(args.outdir, vars(args))
    outdir = Path(args.outdir)
    with open(outdir / "args.json", "w") as f:
        args.wandb_run_id = run_logger.id
        d = vars(args)
        json.dump(d, f, indent=2)
        print(d, flush=True)
    symlink_target = outdir.parent / run_logger.id
    os.symlink(outdir.relative_to(outdir.parent), symlink_target)

    # boilerplate/setup is done
    prompt = get_prompt(args.prompt)
    model = get_model(prompt, args)
    logging.info(f"Example of the used prompt {model.prompt.__post_init__()}")

    if args.test_task2:
        # Predicting scores for the test set (no evaluation is possible without the gold annotations)
        logging.info("Loading task2 test set")
        if args.eval_type == "en":
            logging.warning(
                "The test set is evaluated on paraphrases so the score may be vastly different. Consider also running the same experiment with '--eval_type par' to get score on paraphrases on DEV SET. Submission will still the same."
            )
        assert (
            args.eval_level == "turn"
        ), "We do not support dialogue level evaluation for task2 in this script. Create a new for that"
        task2_df, task2_context_list, task2_response_list, task2_gold_qualities = load_task2_test_dataset(
            insert_speaker=args.insert_speaker, max_examples=args.max_test_examples, 
        )
        logging.info("Predicting scores for task2 test set")
        task2_predicted_scores, task2_total_errors, task2_embeddings = predict(
            "task2",
            model,
            prompt,
            task2_df,
            task2_context_list,
            task2_response_list,
            args,
            gold_qualities=task2_gold_qualities,
        )
        for quality, quality_scores in task2_predicted_scores.items():
            task2_df[quality] = quality_scores  # store to DF
        results_to_csv(task2_df, args.outdir, results_name="task2_test")
        results_to_wandb(task2_df, run_logger, results_name="task2_test")
        if args.save_embeddings:
            task2_outpath = outdir / "task2_embeddings.pth"
            print(f"Embeddings task2 path {str(task2_outpath)}", flush=True)
            torch.save(task2_embeddings, task2_outpath)
        evaluate_metric(
            f"test_task2_turn",
            task2_predicted_scores,
            task2_gold_qualities,
            logger=run_logger,
            metric_qualities=prompt.qualities,
            prefix="test_task2"
        )

    if args.dev_set is not None:
        # predicting and evaluating dev set
        logging.info(f"Loading dev {args.dev_set}")
        df, context_list, response_list, gold_qualities = load_dataset(
            args.dev_set, args.eval_type, insert_speaker=args.insert_speaker
        )
        logging.info(
            f"The dataset {args.dev_set} is annotated with {gold_qualities.keys()}"
        )
        logging.info(f"Predicting scores for {args.dev_set} ")
        predicted_scores, total_errors, embeddings = predict(
            args.dev_set,
            model,
            prompt,
            df,
            context_list,
            response_list,
            args,
            gold_qualities=gold_qualities,
        )
        turns = len(predicted_scores[prompt.qualities[0]])
        msg = f"Total errors: {total_errors} for {turns} turns"
        print(msg, flush=True)
        logging.info(msg, flush=True)
        run_logger.log(dict((f"total_errors_{k}", v) for k, v in total_errors.items()))
        for quality, quality_scores in predicted_scores.items():
            df[quality] = quality_scores  # store to DF
        if args.save_embeddings:
            outpath = outdir / "embeddings.pth"
            print(f"Embeddings devpath {str(outpath)}", flush=True)
            torch.save(embeddings, outpath)
        results_to_csv(df, args.outdir)
        results_to_wandb(df, run_logger)
        evaluate_metric(
            f"{args.dev_set}_{args.eval_type}",
            predicted_scores,
            gold_qualities,
            logger=run_logger,
            metric_qualities=prompt.qualities,
        )

        # Generating train embeddings and computing quality scores on the whole train set
        train_examples = []
        banned_ctxs = [ctx for ctx in df["context"]]
        all_golds = defaultdict(list)  # echt gold
        all_predicted_scores = defaultdict(list)
        all_embeddings = []
        # TODO at least assert that we assume all qualities are in all training datasets
        for dataset in tqdm.tqdm(args.train_sets.strip().split(",")):
            if len(dataset) == 0:
                continue  # skip arg which returns [""]
            logging.info(f"Creating embeddings from {dataset}.")
            df, context_list, response_list, gold_qualities = load_dataset(
                dataset, args.eval_type, banned_ctxs=banned_ctxs
            )
            predicted_scores, total_errors, embeddings = predict(
                dataset,
                model,
                prompt,
                df,
                context_list,
                response_list,
                args,
                gold_qualities=gold_qualities,
            )
            turns = len(predicted_scores[prompt.qualities[0]])
            msg = f"Total errors: {total_errors} for {turns} turns"
            print(msg, flush=True)
            logging.info(msg, flush=True)
            for q in gold_qualities:
                all_golds[q].extend(gold_qualities[q])
            for quality, quality_scores in predicted_scores.items():
                all_predicted_scores[quality].extend(quality_scores)
            all_embeddings.extend(embeddings)
        if args.save_embeddings:
            train_outpath = outdir / "train_embeddings.pth"
            print(f"Embeddings train path {str(train_outpath)}", flush=True)
            torch.save(all_embeddings, train_outpath)
        if len(all_predicted_scores) > 0:  # amount of keys
            evaluate_metric(
                f"train_{args.train_sets}_{args.eval_type}",
                all_predicted_scores,
                all_golds,
                logger=run_logger,
                metric_qualities=prompt.qualities,
                prefix=f"train_{args.train_sets}",
            )

    run_logger.finish()
    print(f"OUTDIR {symlink_target}", flush=True)
    if args.test_task1:
        raise NotImplemented(
            f"take the code from test_task2, create function out of it and apply it for both task1 and task2"
        )
