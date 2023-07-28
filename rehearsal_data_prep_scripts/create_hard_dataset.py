import argparse
import csv
import pandas as pd
import json
from chateval.datasets import load_dataset as load_chateval_dataset

ID = "dialogue_id"
HARD_DEV_PATH = "/lnet/express/projects/semetric/data/dstc11-tr4-chateval/DSTC_11_Track_4/task2/hard-dev/"
RANK_DIR = "/lnet/express/projects/semetric/data/dstc11-tr4-chateval/DSTC_11_Track_4/task2/ranks/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of dataset for creating a hard set"
    )

    parser.add_argument("--eval_type", type=str, default="en")
    parser.add_argument("--ranking_filename", type=str, required=True,
            help="Name of the ranked csv file in the ranks directory")
    parser.add_argument("--out_size", type=int, default=200, help="How many hardest examples to take, default is 200")
    args = parser.parse_args()


    df, _, _, _ = load_chateval_dataset(args.dataset, args.eval_type)
    
    ranking_path = RANK_DIR + args.ranking_filename
    ranked = pd.read_csv(ranking_path)
    ids = ranked[ID].head(args.out_size)

   
    print("Proportion of response sources in the worst ranked subset:")
    print(ranked.head(args.out_size).value_counts(subset=['model'], normalize=True))

    hard_subset = df.loc[df[ID].isin(ids)]
    
    # Now the problem is that the annotations are flattened, so we need to unflatten them
    hard_json = df.to_json(orient="records")
    hard_parsed = json.loads(hard_json)

    o = []
    for r in hard_parsed:
        r["annotations"] = {}
        for k, v in list(r.items()):
            if k.startswith("annotations."):
                ann = k.split('.')[1]
                r["annotations"][ann] = [v]
                del r[k]
            
    with open("{}/{}-hard.json".format(HARD_DEV_PATH, args.dataset), "w") as outfile:
        json.dump(hard_parsed, outfile, indent=4, ensure_ascii=False)
