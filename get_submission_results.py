"""
Tested with 

##### The following example
$ for results in LLM/exp/./run.py-2023-07-27_100046-tdll-3gpu2-34203/task2_test.csv ; do echo; echo $results ; python get_submission_results.py --submission_data $results --subtask t2t --limit_dims APPROPRIATENESS ; echo $results; echo ; done

LLM/exp/./run.py-2023-07-27_100046-tdll-3gpu2-34203/task2_test.csv
Subtask: t2t
Dataset:
Language:
Supervised:

{
    "Spearman": {
        "APPROPRIATENESS": 0.36485201494835806
    },
    "Pearsonr": {
        "APPROPRIATENESS": 0.382746482120679
    }
}
LLM/exp/./run.py-2023-07-27_100046-tdll-3gpu2-34203/task2_test.csv

##### end of example

"""
import argparse
import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def spearman_pearson(dstc11_dim_list, df_participant_dim_list):
    # Dimension with annotations
    try:
        spearman_corr = abs(spearmanr(dstc11_dim_list, df_participant_dim_list)[0])
        pearson_corr = abs(pearsonr(dstc11_dim_list, df_participant_dim_list)[0])
    # Dimension without annotations
    except:
        spearman_corr = '-'
        pearson_corr = '-'

    return spearman_corr, pearson_corr

def compute_metrics(args):

    df_participant_results = pd.read_csv(args.submission_data)
    subtask = args.subtask # 't1t', 't1d', 't2t', 't2d'
    dataset = args.dataset # '', 'DSTC10PERSONA', 'DSTC10TOPICAL', 'ESL', 'JSALT', 'NCM', 'HCCHNS', 'BLENDERBOT3', 'CHATGPT', 'GPT3'
    language = args.language # '', 'EN', 'ZH', 'ES'
    supervised = args.supervised # '', 'YES', 'NO'

    # Load test data results with annotations for a especific subtask
    if subtask == 't1t':
        df_dstc11 = pd.read_csv(args.t1t_data_path)
    elif subtask == 't1d':
        df_dstc11 = pd.read_csv(args.t1d_data_path)
    elif subtask == 't2t':
        df_dstc11 = pd.read_csv(args.t2t_data_path)
    elif subtask == 't2d':
        df_dstc11 = pd.read_csv(args.t2d_data_path)
    
    # Evaluate chatbot turns only
    if subtask == 't1t' or subtask == 't2t':
        df_dstc11 = df_dstc11[df_dstc11['SID'] == 'Chatbot']
    
    # Evaluate a specific dataset
    if dataset:
        indexes = df_dstc11.index[df_dstc11['CID'] == dataset].tolist()
        df_dstc11 = df_dstc11.loc[indexes]
    
    # Evaluate a specific language
    if language and (subtask == 't1t' or subtask == 't1d'):
        indexes = df_dstc11.index[df_dstc11['LANGUAGE'] == language].tolist()
        df_dstc11 = df_dstc11.loc[indexes]

    # Evaluate turns by type of supervision
    if supervised:
        indexes = df_dstc11.index[df_dstc11['SUPERVISED'] == supervised].tolist()
        df_dstc11 = df_dstc11.loc[indexes]
    
    # Turn-level dimensions
    if subtask == 't1t' or subtask == 't2t':
        dim1 = 'APPROPRIATENESS'
        dim2 = 'CONTENT_RICHNESS'
        dim3 = 'GRAMMATICAL_CORRECTNESS'
        dim4 = 'RELEVANCE'
    # Dial-level dimensions
    elif subtask == 't1d' or subtask == 't2d':
        dim1 = 'COHERENCE'
        dim2 = 'ENGAGENESS'
        dim3 = 'INFORMATIVENESS'
        dim4 = 'OVERALL'
    dims = [dim1,dim2,dim3,dim4]
    if args.limit_dims is not None:
        selected_dims = set(args.limit_dims.split(','))
        assert all([sd in dims for sd in selected_dims]), f"{selected_dims=} vs {dims=}"
        dims = [d for d in dims if d in selected_dims]

    
    # Get the indexes of the turns that have annotations in each dimension separately
    spearman_corr_scores, pearsonr_corr_scores = [], []
    for i, dim in enumerate(dims):

        dstc11_dim_index = df_dstc11[df_dstc11[dim].notnull()].index.to_list()

        # Get the test annotations
        dstc11_dim_list = df_dstc11[dim].loc[dstc11_dim_index].to_list()
        
        # Get the participant's annotations
        df_participant_dim_list = df_participant_results[dim].loc[dstc11_dim_index].to_list()
    
        # Calculate Spearman and Peason correlations
        score_spearman, score_pearson = spearman_pearson(dstc11_dim_list, df_participant_dim_list)
        spearman_corr_scores.append(score_spearman)
        pearsonr_corr_scores.append(score_pearson)

    return dims, spearman_corr_scores, pearsonr_corr_scores


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_data', default=None, type=str, required=True, help="Absolute path to participant test data submission")
    parser.add_argument('--subtask', default='', type=str, required=True, help="Subtask of the submission data")
    parser.add_argument('--limit_dims', default=None, type=str)
    parser.add_argument('--dataset', default='', type=str, required=False, help="Evaluate a especific dataset (leave empty to evaluate all the datasets)")
    parser.add_argument('--language', default='', type=str, required=False, help="Evaluate a especific language (leave empty to evaluate all the language)")
    parser.add_argument('--supervised', default='', type=str, required=False, help="Evaluate turns by type of supervision (leave empty to evaluate supervised and not supervised data)")
    parser.add_argument('--t1t_data_path', default='chateval-data/DSTC_11_Track_4/eval/task1/dstc11_test_task1_turn.csv', type=str, required=False, help="Path of Task 1 Turn test data")
    parser.add_argument('--t1d_data_path', default='chateval-data/DSTC_11_Track_4/eval/task1/dstc11_test_task1_dial.csv', type=str, required=False, help="Path of Task 1 Dial test data")
    parser.add_argument('--t2t_data_path', default='chateval-data/DSTC_11_Track_4/eval/task2/dstc11_test_task2_turn.csv', type=str, required=False, help="Path of Task 2 Turn test data")
    parser.add_argument('--t2d_data_path', default='chateval-data/DSTC_11_Track_4/eval/task2/dstc11_test_task2_dial.csv', type=str, required=False, help="Path of Task 2 Dial test data")
    args=parser.parse_args()

    if not args.submission_data:
        raise Exception("Please provide a valid submission_data.")
    dims, scores_spearman, scores_pearsonr = compute_metrics(args)
    
    print('Subtask: ' + args.subtask)
    print('Dataset: ' + args.dataset)
    print('Language: ' + args.language)
    print('Supervised: ' + args.supervised + '\n')

    keys = ["Spearman", "Pearsonr"]
    results = {keys[0]:{}, keys[1]:{}}
    [results[keys[0]].update({dim: scores_spearman[i]}) for i, dim in enumerate(dims)]
    [results[keys[1]].update({dim: scores_pearsonr[i]}) for i, dim in enumerate(dims)]

    # results = {
    #     'Spearman': {
    #         dims[0]: scores_spearman[0],
    #         dims[1]: scores_spearman[1],
    #         dims[2]: scores_spearman[2],
    #         dims[3]: scores_spearman[3]
    #     },
    #     'Pearsonr': {
    #         dims[0]: scores_pearsonr[0],
    #         dims[1]: scores_pearsonr[1],
    #         dims[2]: scores_pearsonr[2],
    #         dims[3]: scores_pearsonr[3]
    #     }
    # }
    
    print(json.dumps(results, indent=4))
