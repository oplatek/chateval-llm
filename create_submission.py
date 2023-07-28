import argparse
import csv
import pandas as pd
import os


DATA_DIR = "chateval-data/DSTC_11_Track_4/submissions"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--appropriateness", type=str, required=True)
    parser.add_argument("--richness", type=str, required=True)
    parser.add_argument("--grammatical", type=str, required=True)
    parser.add_argument("--relevance", type=str, required=True)
    parser.add_argument("--submission_name", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=False)

    args = parser.parse_args()

    outdir = DATA_DIR if args.outdir is None else args.outdir
    OUT_FILE = os.path.join(outdir, "ufal-cuni_task2_turn_v_{}.csv")
    # Create a pandas dataframe with column names "APPROPRIATENESS", "CONTENT_RICHNESS", "GRAMMATICAL_CORRECTNESS", "RELEVANCE"
    results = pd.DataFrame(
        columns=[
            "UID",
            "APPROPRIATENESS",
            "CONTENT_RICHNESS",
            "GRAMMATICAL_CORRECTNESS",
            "RELEVANCE",
        ]
    )

    df = pd.read_csv(args.appropriateness)
    results["UID"] = df["UID"]
    results["APPROPRIATENESS"] = df["appropriateness"]

    df = pd.read_csv(args.richness)
    results["CONTENT_RICHNESS"] = df["richness"]
    assert results["UID"].equals(
        df["UID"]
    ), "Appropriateness file has different UIDs than richness file!"

    df = pd.read_csv(args.grammatical)
    results["GRAMMATICAL_CORRECTNESS"] = df["grammatical"]
    assert results["UID"].equals(
        df["UID"]
    ), "Appropriateness file has different UIDs than grammatical file!"

    df = pd.read_csv(args.relevance)
    results["RELEVANCE"] = df["relevance"]
    assert results["UID"].equals(
        df["UID"]
    ), "Appropriateness file has different UIDs than relevance file!"

    # Perform a check to see if any UIDs end with 0 (first turn and we should not predict those):
    first = [uid for uid in results["UID"] if uid.endswith("0000")]
    if len(first) > 0:
        print("Removing first turns from the submission file")
        results = results[~results["UID"].isin(first)]

    assert len(results) == 1790, (len(results), 1790)
    results.to_csv(OUT_FILE.format(args.submission_name), index=False)
