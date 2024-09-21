import glob
import os.path
from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt
from scipy.stats import hmean

import numpy as np
from scripts.evaluation.statistical_analysis import COLORS

def analyze_re_samplings(path_to_results):
    with open(path_to_results, 'rt') as f:
        results = json.load(f)
    total = {}
    for model in results["models"]:
        res_by_id = {}
        model_ids = results["models"][model]["ids"]
        model_scores = results["models"][model]["scores"]
        for list_index in range(len(model_ids)):
            scores_for_list_index = model_scores[list_index]
            metrics = scores_for_list_index.keys()
            scores_for_iteration = np.array([scores_for_list_index[k] for k in scores_for_list_index])
            if 'rouge' in list(metrics)[0].lower():
                # geometric mean for rouge
                mean = np.prod(scores_for_iteration, axis=0) ** (1 / len(metrics)) * 100
            elif 'faithfulness' in list(metrics)[0].lower():
                mean = hmean(scores_for_iteration, axis=0)
            else:
                mean = scores_for_iteration.squeeze()
            for i, sample_index in enumerate(model_ids[list_index]):
                if sample_index not in res_by_id:
                    res_by_id[sample_index] = []
                res_by_id[sample_index].append(mean[i])
        acc_res = {}
        for sample_id in res_by_id:
            scores_list = res_by_id[sample_id]
            min_k = np.min(scores_list)
            max_k = np.max(scores_list)
            mean_k = np.mean(scores_list)
            std = np.std(scores_list)
            acc_res[sample_id] = {"min": min_k, "max": max_k, "mean": mean_k, "std": std}
        all_means = np.array([acc_res[res]["mean"] for res in acc_res])
        all_stds = np.array([acc_res[res]["std"] for res in acc_res])
        percentile_mean = np.percentile(all_means, q=20)
        percentile_std = np.percentile(all_stds, q=50)
        seemingly_hard = [sample_id for sample_id in acc_res if acc_res[sample_id]["mean"] <= percentile_mean]
        hard_samples = [sample_id for sample_id in seemingly_hard if acc_res[sample_id]["std"] <= percentile_std]
        # sensitive_samples = [sample_id for sample_id in acc_res if acc_res[sample_id]["std"] >= np.percentile(all_stds, q=80)]
        # low_means = [sample_id for sample_id in acc_res if acc_res[sample_id]["mean"] <= percentile_mean]
        # not_hard = [s for s in low_means if s in sensitive_samples]
        total[model] = {"mean_thresh_low": percentile_mean,
                        "std_lower": percentile_std,
                        "std_upper": np.percentile(all_stds, q=80),
                        "total_samples_num": len(all_means),
                        "20% num": int(len(all_means)*0.2),
                        "num_hard": len(hard_samples),
                        # "num_sensitive": len(sensitive_samples),
                        # "num_not_hard": len(not_hard),
                        "hard": hard_samples,
                        "seemingly_hard": seemingly_hard,
                        "precentage_hard": len(hard_samples)/len(seemingly_hard)
                        # "sensitive": sensitive_samples,
                        # "not_hard": not_hard
                        }
    results_dir_path = os.path.dirname(path_to_results)
    path_to_analyze = os.path.join(results_dir_path, "hard_vs_sensitive.json")
    with open(path_to_analyze, 'wt') as f:
        str_json = json.dumps(total, indent=2)
        f.write(str_json)

    print(total)
    # COLORS.reverse()
    # Extract keys and values
    models = ["Gemma1.1-2B", "Gemma1.1-7B", "Llama3-8B", "Llama3-70B", "Mistral-7B", "Mixtral-8x7B", "Mixtral-8x22B"]
    values = []
    for model in models:
        hard = int(total[model]["precentage_hard"] * 100)
        not_hard = 100 - hard
        values.append([hard, not_hard])
    print(values)




    values = list(np.array([len(total[model]["hard"]) for model in models])/int(len(all_means)*0.2))
    out_dir = os.path.dirname(path_to_results)
    path_to_sensitive_fig = os.path.join(out_dir, "hard_vs_sensitive.pdf")
    path_to_hard = os.path.join(out_dir, "hard_examples.pdf")

    # Create horizontal bar chart
    plt.barh(models, values, color=COLORS)
    plt.xlabel("Percentage of actual\nhard examples", fontdict={"size": 16})
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(path_to_hard)

    values = list(np.array([len(total[model]["not_hard"]) for model in models]) / int(len(all_means) * 0.2))
    # Create horizontal bar chart
    plt.barh(models, values, color=COLORS)
    plt.xlabel("Percentage of ''hard'' examples that are\nactually sensitive", fontdict={"size": 16})
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.savefig(path_to_sensitive_fig, bbox_inches='tight')






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_path", required=True,
                        help="path to results json")
    args = parser.parse_args()
    analyze_re_samplings(args.results_path)


