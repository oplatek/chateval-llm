import os
import numpy as np
import socket
from typing import Dict, Any
import wandb
import csv
import datetime
import logging
import re
from scipy.stats import spearmanr, pearsonr
import pandas as pd


"""
We measure the quality of the response in four dimensions:

Appropriateness - The response is appropriate given the preceding dialogue.
Content Richness - The response is informative, with long sentences including multiple entities and conceptual or emotional words.
Grammatical Correctness - Responses are free of grammatical and semantic errors.
Relevance - Responses are on-topic with the immediate dialog history.
"""
TURN_LEVEL_METRICS = ("appropriateness", "richness", "grammatical", "relevance")


def exp_dirname(args, main_file=None):
    script_file = main_file or globals().get("__file__", "notebook")
    return os.path.join(
        "exp",
        f"{os.path.basename(script_file)}-{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}-{socket.gethostname()}-{os.getpid()}",
    )


def get_wb_run_logger(outdir, hyperparams: Dict[str, Any]):
    # crucial parameters - pinned columns
    assert "prompt" in hyperparams
    assert "model_name" in hyperparams
    return wandb.init(
        entity="metric",
        project="chateval",
        dir=outdir,
        config=hyperparams,
        settings=wandb.Settings(code_dir="."),
    )


def evaluate_metric(
    eval_type, scores, human_scores, metric_qualities=None, logger=None, prefix=None
):
    prefix = "" if prefix is None else f"{prefix.rstrip('-')}-"
    metric_qualities = (
        TURN_LEVEL_METRICS if metric_qualities is None else metric_qualities
    )
    pear_sum = 0.0
    spear_sum = 0.0
    count = 0
    for quality in metric_qualities:
        hs = human_scores.get(f"annotations.{quality}")
        if hs is None:
            logging.warning(
                f"{eval_type} does not have annotation for {quality}. Skipping"
            )
            continue
        
        logging.info(f"{eval_type} evaluating {prefix}{quality=}")
        ps = scores[quality]

        if not isinstance(hs, np.ndarray):
            hs = np.array(hs)
        if not isinstance(ps, np.ndarray):
            ps = np.array(ps)

        # Note that nan values are returned if ps is constant - e.g. predict always 1.0

        # filter values for which the human score is not available or is nonsensical
        mask = ~np.isnan(hs)
        hs = hs[mask]
        ps = ps[mask]

        pear, p = pearsonr(hs, ps)
        logging.info(
            f"{eval_type} -- Pearson:  {prefix}{quality} => {pear:6.3f}  p={p:6.4f}"
        )
        spear, p = spearmanr(hs, ps)
        logging.info(
            f"{eval_type} -- Spearman: {prefix}{quality} => {spear:6.3f} p={p:6.4f}"
        )
        pear_sum += pear
        spear_sum += spear
        count += 1
        if logger is not None:
            logger.log(
                {
                    f"{prefix}PCC-{quality}": pear,
                    f"{prefix}SRCC-{quality}": spear,
                }
            )

    pear_mean = pear_sum / count
    logging.info(f"{eval_type} Pearson mean r: {pear_mean:6.4f}")
    spear_mean = spear_sum / count
    logging.info(f"{eval_type} Spearman mean r: {spear_mean:6.4f}")
    if logger is not None:
        logger.log(
            {
                f"{prefix}PCC-mean": pear_mean,
                f"{prefix}SRCC-mean": spear_mean,
            }
        )
    return pear_mean, spear_mean


def results_to_csv(df: pd.DataFrame, outdir: str, results_name=None):
    results_name = results_name or "results"
    df.to_csv(
        f"{outdir}/{results_name}.csv",
        index=None,
        lineterminator="\n",
        quoting=csv.QUOTE_NONNUMERIC,
    )


def results_to_wandb(df: pd.DataFrame, logger, results_name=None):
    results_name = results_name or "results"
    logger.log({results_name: wandb.Table(dataframe=df)})
