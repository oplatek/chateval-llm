from collections import defaultdict
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from pandas import DataFrame


CHATEVAL_ROOT = Path(__file__).parent.parent
DATA_DIR = CHATEVAL_ROOT / "chateval-data"
DATASET_TEMPLATE = (
    "DSTC_11_Track_4/metadata/dev/en/{dataset}/{dataset}_eval_zh_es_pa.json"
)
# Datasets should be named XY-hard if this template should be used
HARD_TEMPLATE = "DSTC_11_Track_4/task2/hard-dev/{dataset}.json"
EVAL_DIR = DATA_DIR / "DSTC_11_Track_4" / "eval"


def _insert_speaker(ctx):
    turns = ctx.strip().split("\n")
    turns = list(
        reversed(
            [
                f"A: {t}" if i % 2 == 0 else f"B: {t}"
                for i, t in enumerate(reversed(turns))
            ]
        )
    )
    return "\n".join(turns)


def load_task2_test_dataset(insert_speaker=False, max_examples=None):
    """Assumes the csv is sorted bysed on UID first based on dialogue ID than according to turn number"""
    # Columns for task2 UID,SID,PARAPHRASES ... for task1 they are UID,SID,SEG
    # First column is turn id inf format TEST-DIALOGUEID-TURNID
    # Headers: UID,SID,PARAPHRASES
    path_dataset = EVAL_DIR / "task2/dstc11_paraphrases_test.csv"
    # Headers: UID,UID_TEST,CID,SID,SUPERVISED,APPROPRIATENESS,CONTENT_RICHNESS,GRAMMATICAL_CORRECTNESS,RELEVANCE
    annotation_path_dataset = EVAL_DIR / "task2/dstc11_test_task2_turn.csv"

    quality_mapping = {
        "APPROPRIATENESS": "annotations.appropriateness",
        "CONTENT_RICHNESS": "annotations.richness",
        "GRAMMATICAL_CORRECTNESS": "annotations.grammatical",
        "RELEVANCE": "annotations.relevance",
    }
    ann_df = pd.read_csv(annotation_path_dataset)

    df = pd.read_csv(path_dataset)
    df = df.loc[~df["UID"].str.endswith("000", na=False)]
    response_list = df.PARAPHRASES.to_list()
    turn_ids = df.UID.to_list()
    d = defaultdict(list)
    context_list = []
    qualities = dict((q_ourname, []) for q_ourname in quality_mapping.values())
    for i, (tid, r) in enumerate(zip(turn_ids, response_list)):
        if max_examples is not None and i >= max_examples:
            break
        TEST_STR, dialogue_id, turn_id = tid.split("-")
        ann_row = ann_df[ann_df["UID_TEST"] == tid] 

        for q_row, q_ourname in quality_mapping.items():
            qualities[q_ourname].append(ann_row[q_row].values[0])
        assert TEST_STR == "TEST", TEST_STR
        # dialogue context is all previous turns without the current turn
        context = "\n".join(d[dialogue_id])
        if insert_speaker:
            context = _insert_speaker(context)
        context_list.append(context)
        # adding the current turn as it will be context for the next turns
        d[dialogue_id].append(r.strip())
    if max_examples is not None:
        df = df.head(max_examples)
    df["context"] = context_list
    return df, context_list, response_list, qualities


def load_dataset(
    dataset: str, eval_type, max_turns=-1, banned_ctxs=None, insert_speaker=False
):
    if banned_ctxs is None:
        banned_ctxs = []
    if dataset.endswith("-hard"):
        path_dataset = DATA_DIR / HARD_TEMPLATE.format(dataset=dataset)
    else:
        path_dataset = DATA_DIR / DATASET_TEMPLATE.format(dataset=dataset)
    logging.info(f"loading {path_dataset}")
    with open(path_dataset) as f:
        df = pd.json_normalize(json.load(f))
    df = normalize_df(dataset, df, dataset_meta_info)
    df = df.loc[~df["context"].isin(banned_ctxs)]

    if eval_type == "en":
        response_list = df.response.to_list()
        ct_list = df.context.to_list()
    elif eval_type == "zh":
        response_list = df.response_zh.to_list()
        ct_list = df.context_zh.to_list()
    elif eval_type == "es":
        response_list = df.response_es.to_list()
        ct_list = df.context_es.to_list()
    elif eval_type == "par":
        response_list = df.response_pa.to_list()
        ct_list = df.context_pa.to_list()
    else:
        raise ValueError("Please, specify a valid type of evaluation: en, zh, es, par")

    response_list = [item if item != "" else "no response" for item in response_list]

    if insert_speaker:
        context_list = [_insert_speaker(item) for item in ct_list]
        response_role = "B: "
    else:
        context_list = [item.strip() for item in ct_list]
        response_role = ""
    response_list = [f"{response_role}{item.strip()}" for item in response_list]
    logging.info(context_list[:2])
    logging.info(response_list[:2])
    logging.info(f"{len(context_list)=} {len(response_list)=}")
    annotations = [
        "annotations." + _ for _ in dataset_meta_info[dataset]["annotations"]
    ]
    human_scores = {}
    for k in annotations:
        human_scores[k] = list(df[k])

    if max_turns > 1:
        df = df.head(max_turns)
        context_list = context_list[:max_turns]
        response_list = response_list[:max_turns]
        for v in human_scores.values():
            v = v[:max_turns]

    qualities = scores2test_qualities(human_scores, dataset)

    return df, context_list, response_list, qualities


def scores2test_qualities(human_scores, dataset):
    # TODO refactor directly into chateval.datasets.load_dataset
    from chateval.results import TURN_LEVEL_METRICS

    metrics_mapping = turn_metric_mapping[dataset]
    human_metric_scores = {}
    for quality in metrics_mapping:
        relevant_columns = np.array(
            [
                human_scores[f"annotations.{source_metric}"]
                for source_metric in metrics_mapping[quality]
            ]
        )
        human_metric_scores[f"annotations.{quality}"] = np.mean(
            relevant_columns, axis=0
        )
    return human_metric_scores


def normalize_df(dataset_name, df, dataset_meta_info):
    dataset_meta = dataset_meta_info[dataset_name]
    for annotation in dataset_meta["annotations"]:
        df["annotations." + annotation] = df["annotations." + annotation].apply(
            dataset_meta["aggregation"]
        )
    return df


dataset_meta_info = {
    "convai2-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    "dailydialog-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    "dailydialog-gupta": {"annotations": ["overall"], "aggregation": lambda x: x[0]},
    "dailydialog-predictive": {"annotations": ["overall"], "aggregation": np.mean},
    "dailydialog-holistic": {
        "annotations": ["relevance"],
        "aggregation": lambda x: x[0],
    },
    "dailydialog-zhao": {
        "num_references": 1,
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dailydialog-zhao-hard": {
        "num_references": 1,
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "dailydialog-zhao-3": {
        "num_references": 1,
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    # "dstc6": {"num_references": 11, "annotations": ["overall"], "aggregation": np.mean},
    "dstc7": {
        "num_references": 1,
        "annotations": ["relevance", "informativeness", "overall"],
        "aggregation": np.mean,
    },
    # "dstc10-persona": {
    #     "annotations": ["content", "grammar", "appropriateness", "relevance"],
    #     "aggregation": np.mean,
    # },
    # "dstc10-topical": {
    #     "annotations": ["content", "grammar", "appropriateness", "relevance"],
    #     "aggregation": np.mean,
    # },
    "empathetic-grade": {"annotations": ["relevance"], "aggregation": np.mean},
    # "esl": {"annotations": ["appropriateness"], "aggregation": lambda x: x[0]},
    "fed-turn": {
        "annotations": [
            "Correct",
            "Engaging",
            "Fluent",
            "Interesting",
            "Overall",
            "Relevant",
            "Semantically appropriate",
            "Specific",
            "Understandable",
        ],
        "aggregation": np.mean,
    },
    "fed-turn-hard": {
        "annotations": [
            "Correct",
            "Engaging",
            "Fluent",
            "Interesting",
            "Overall",
            "Relevant",
            "Semantically appropriate",
            "Specific",
            "Understandable",
        ],
        "aggregation": np.mean,
    },
    "fed-dial": {
        "annotations": [
            "Coherent",
            "Error recovery",
            "Consistent",
            "Diverse",
            "Depth",
            "Likeable",
            "Understanding",
            "Flexible",
            "Informative",
            "Inquisitive",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "humod": {
        "num_references": 3,
        "annotations": ["language_usage", "relevance"],
        "aggregation": np.mean,
    },
    # "jsalt": {"annotations": ["appropriateness"], "aggregation": np.mean},
    # "ncm": {"annotations": ["appropriateness"], "aggregation": lambda x: x[0]},
    "persona-see": {
        "annotations": [
            "avoid_rep",
            "enjoy",
            "fluency",
            "inquisitive",
            "interest",
            "listen",
            "make_sense",
            "persona_guess",
            "turing",
        ],
        "aggregation": lambda x: x[0],
    },
    "persona-see-hard": {
        "annotations": [
            "avoid_rep",
            "enjoy",
            "fluency",
            "inquisitive",
            "interest",
            "listen",
            "make_sense",
            "persona_guess",
            "turing",
        ],
        "aggregation": lambda x: x[0],
    },
    "persona-usr": {
        "num_references": 1,
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "persona-usr-hard": {
        "num_references": 1,
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "persona-zhao": {
        "num_references": 1,
        "annotations": ["appropriateness"],
        "aggregation": np.mean,
    },
    "rehearsal-copy-hard": {
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "rehearsal-constant-hard": {
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "rehearsal-omit-hard": {
        "annotations": ["content", "grammar", "appropriateness", "relevance"],
        "aggregation": np.mean,
    },
    "topical-usr": {
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
    "topical-usr-hard": {
        "annotations": [
            "Understandable",
            "Natural",
            "Maintains Context",
            "Engaging",
            "Uses Knowledge",
            "Overall",
        ],
        "aggregation": np.mean,
    },
}


# See results.TURN_LEVEL_METRICS
turn_metric_mapping = {
    "convai2-grade": {"relevance": ["relevance"]},
    "dailydialog-grade": {"relevance": ["relevance"]},
    "dailydialog-gupta": {"relevance": ["overall"]},
    # todo investigate
    "dailydialog-predictive": {},
    "dailydialog-holistic": {"relevance": ["relevance"]},
    "dailydialog-zhao": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "dailydialog-zhao-hard": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "dailydialog-zhao-3": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "dstc7": {
        "relevance": ["relevance"],
        "richness": ["informativeness"],
    },
    "empathetic-grade": {
        "relevance": ["relevance"],
    },
    # "esl": {"annotations": ["appropriateness"], "aggregation": lambda x: x[0]},
    "fed-turn": {
        "relevance": ["Relevant"],
        "appropriateness": ["Semantically appropriate", "Correct"],
        "richness": ["Engaging", "Interesting", "Specific"],
        "grammatical": ["Understandable", "Fluent"],
    },
    "fed-turn-hard": {
        "relevance": ["Relevant"],
        "appropriateness": ["Semantically appropriate", "Correct"],
        "richness": ["Engaging", "Interesting", "Specific"],
        "grammatical": ["Understandable", "Fluent"],
    },
    "fed-dial": {
        # Should we use dialogue level labels for turns?
        # "relevance": ["Depth", "Consistent", "Error recovery"],
        # "appropriateness": ["Understanding", "Likeable", "Consistent", "Flexible"],
        # "richness": ["Depth", "Informative", "Diverse"],
        # "grammatical": ["Coherent"],
    },
    "humod": {
        "relevance": ["relevance"],
        "grammatical": ["language_usage"],
    },
    # These labels are vague
    "persona-see": {
        "relevance": ["make_sense"],
        "appropriateness": ["listen", "make_sense"],
        "richness": ["avoid_rep", "enjoy", "fluency", "enjoy", "inquisitive"],
        "grammatical": ["fluency"],
    },
    "persona-see-hard": {
        "relevance": ["make_sense"],
        "appropriateness": ["listen", "make_sense"],
        "richness": ["avoid_rep", "enjoy", "fluency", "enjoy", "inquisitive"],
        "grammatical": ["fluency"],
    },
    # TODO-uses knowledge 0-1
    # TODO-understandable 0-1
    "persona-usr": {
        "relevance": ["Understandable"],
        "appropriateness": ["Maintains Context", "Uses Knowledge", "Understandable"],
        "richness": ["Uses Knowledge", "Engaging"],
        "grammatical": ["Natural", "Understandable"],
    },
    "persona-usr-hard": {
        "relevance": ["Understandable"],
        "appropriateness": ["Maintains Context", "Uses Knowledge", "Understandable"],
        "richness": ["Uses Knowledge", "Engaging"],
        "grammatical": ["Natural", "Understandable"],
    },
    "persona-zhao": {
        "appropriateness": ["appropriateness"],
    },
    "rehearsal-copy-hard": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "rehearsal-constant-hard": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "rehearsal-omit-hard": {
        "relevance": ["relevance"],
        "appropriateness": ["appropriateness"],
        "richness": ["content"],
        "grammatical": ["grammar"],
    },
    "topical-usr": {
        "relevance": ["Maintains Context", "Uses Knowledge"],
        "richness": ["Natural", "Engaging"],
        "grammatical": ["Understandable"],
    },
    "topical-usr-hard": {
        "relevance": ["Maintains Context", "Uses Knowledge"],
        "richness": ["Natural", "Engaging"],
        "grammatical": ["Understandable"],
    },
}
