import argparse
import csv
import json
import pandas as pd

HARD_DEV_PATH = "/lnet/express/projects/semetric/data/dstc11-tr4-chateval/DSTC_11_Track_4/task2/hard-dev/"
# Column mapping
ID = 0
SID = 1
RESPONSE = 2
P_APPR = 3
P_CONT = 4
P_GRAM = 5
P_REL = 6
O_APPR = 7
O_CONT = 8
O_GRAM = 9
O_REL = 10

def handle_beginning_of_dialogue(context, line, df, type):
    if type == "constant":
        return add_row(context, line, df, override_rel=5.0)
    elif type == "copy":
        appr = (float(line[P_APPR]) + float(line[O_APPR])) / 2
        return add_row(context, line, df, override_rel=appr)
    elif type == "omit":
        return df

def resolve_qualities(line, override_rel=None):
    if line[O_CONT] is not None:
        cont = (float(line[P_CONT]) + float(line[O_CONT])) / 2
        gram = (float(line[P_GRAM]) + float(line[O_GRAM])) / 2
        appr = (float(line[P_APPR]) + float(line[O_APPR])) / 2
        if override_rel is not None:
            rel = override_rel
        else:
            rel = (float(line[P_REL]) + float(line[O_REL])) / 2
    else:
        cont = float(line[P_CONT])
        gram = float(line[P_GRAM])
        appr = float(line[P_APPR])
        if override_rel is not None:
            rel = override_rel
        else:
            rel = float(line[P_REL])

    return cont, gram, appr, rel

def add_row(context, line, df, override_rel=None):
    cont, gram, appr, rel = resolve_qualities(line, override_rel)
    row = pd.DataFrame([[line[ID], line[SID], context, line[RESPONSE], cont, gram, appr, rel]],
                       columns=["dialogue_id", "model", "context", "response", "annotations.content",
                                "annotations.grammar", "annotations.appropriateness", "annotations.relevance"])
    return pd.concat([df, row], ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True) # Input csv file
    parser.add_argument("--type", type=str, default="constant", choices=["constant", "copy", "omit"])

    args = parser.parse_args()

    # Create an empty pandas dataframe with column names dialogue_id, model, context, response, annotations.content, annotations.grammar, annotations.appropriateness, annotations.relevance
    df = pd.DataFrame(columns=["dialogue_id", "model", "context", "response", "annotations.content", "annotations.grammar", "annotations.appropriateness", "annotations.relevance"])

    with open(args.input, "r") as infile:
        # Read csv file line by line, skip the header
        reader = csv.reader(infile)
        next(reader)
        context = ""
        for line in reader:
            # Turn ID is the last part of the dialogue ID
            turn_id = line[ID].split("-")[-1]
            # If it ends with 0, it's the beginning of a new dialogue
            if turn_id.endswith("0"):
                context = ""
                df = handle_beginning_of_dialogue(context, line, df, args.type)
                context = line[RESPONSE]
            else:
                df = add_row(context, line, df)
                context += '\n' + line[RESPONSE]


    # Add a column response_zh containing empty strings to df
    df["response_zh"] = ""
    df["response_es"] = ""
    df["response_pa"] = ""
    # Now unflatten the annotations
    hard_json = df.to_json(orient="records")
    hard_parsed = json.loads(hard_json)

    for r in hard_parsed:
        r["annotations"] = {}
        for k, v in list(r.items()):
            if k.startswith("annotations."):
                ann = k.split('.')[1]
                r["annotations"][ann] = [v]
                del r[k]

    with open("{}/rehearsal-{}-hard.json".format(HARD_DEV_PATH, args.type), "w") as outfile:
        json.dump(hard_parsed, outfile, indent=4, ensure_ascii=False)







