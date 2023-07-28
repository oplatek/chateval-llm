#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import wandb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import glob
from itertools import product

MEAN_DIFF = "mean_diff"
OUT_DIR = "/lnet/express/projects/semetric/data/dstc11-tr4-chateval/DSTC_11_Track_4/task2/ranks/"

# Key from gold column names and values from predicted
quality_mapping = {
        "relevance": "relevance",
        "appropriateness": "appropriateness",
        "content": "richness",
        "grammar": "grammatical"}

def get_wandb_table(run_name, table_name):
    api = wandb.Api()
    art = api.artifact(run_name)
    table = art.get(table_name)
    return pd.DataFrame(data=table.data, columns=table.columns)


def get_table_from_exp(run_id):
    csv_path = '../chateval-exp/{}/*csv'.format(run_id)
    csv_path = glob.glob(csv_path)
    if len(csv_path) < 1:
        print('Run not found, check the symlink')
        return None
    return pd.read_csv(csv_path[0])

# Ideal use case when we have predictions for each category separately)
def get_relevant_pair_rankings(gold, predicted, df):
    rank_diff_columns = []
    for g in gold:
        g_name = g.split('.')[1]
        pred_candidates = [p for p in predicted if p.endswith(quality_mapping[g_name])]
        if len(pred_candidates) < 1:
            print("Could not find gold quality {} in predicted data as {}".format(g_name, quality_mapping[g_name]))
        if len(pred_candidates) > 1:
            print("Found more predicted quality columns ending with {} corresponding to gold quality {}".format(quality_mapping[g_name], g_name))
        else:
            p = pred_candidates[0]
            name = "{}_vs_{}".format(g_name, p)
            df[name] = (df["{}_rank".format(g)] - df["{}_rank".format(p)]).abs()
            rank_diff_columns.append(name)
    df[MEAN_DIFF] = df[rank_diff_columns].mean(axis=1)
    return df


def get_carthesian_prod_diffs(gold, predicted, df):
    rank_diff_columns = []
    for g,p in product(gold, predicted):
        name = "{}_vs_{}".format(g,p)
        df[name] = (df["{}_rank".format(g)] - df["{}_rank".format(p)]).abs()
        rank_diff_columns.append(name)
    df[MEAN_DIFF] = df[rank_diff_columns].mean(axis=1)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--run_id",
            type = str,
            required = True,
            help = "Can be found in URL, looks like a random string")
    parser.add_argument(
            "--dataset",
            type = str,
            required = True,
            help = "The dataset to be ranked")
    parser.add_argument(
            "--metric",
            type = str,
            required = True,
            help = "The metric that was used to predict qualities")
    args = parser.parse_args()


    #table = get_wandb_table(project + run_name, table_name)
    table = get_table_from_exp(args.run_id)
    
    gold_qualities = [q for q in table.columns if q.startswith('annotation')]
    if 'am' in args.metric.lower():
        predicted_qualities = [q for q in table.columns if q.endswith('scores')]
    else:
        predicted_qualities = [q for q in table.columns if q.startswith('LLM')]

    df = pd.DataFrame(list(zip(table['dialogue_id'], table['model'])), columns=['dialogue_id', 'model'])
    
    # Add absolute values of the qualities to the dataframe
    for q in gold_qualities:
        df[q] = table[q]
        df["{}_rank".format(q)] = rankdata(table[q])
    for q in predicted_qualities:
        df[q] = table[q]
        df["{}_rank".format(q)] = rankdata(table[q])

    if len(gold_qualities) > len(predicted_qualities):
        df = get_carthesian_prod_diffs(gold_qualities, predicted_qualities, df)  
    elif len(gold_qualities) == len(predicted_qualities):
        df = get_relevant_pair_rankings(gold_qualities, predicted_qualities, df)
        

    df = df.sort_values(by=[MEAN_DIFF], ascending=False)
    df.to_csv('{}{}_{}_ranks_sorted.csv'.format(OUT_DIR, args.dataset, args.metric))
    print("Ranking saved as '{}_{}_ranks_sorted.csv'".format(args.dataset, args.metric))




