# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde







def plot_distributions(original_crit, rewritten_crit, bandwidth=0.3):
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")  # 设置seaborn风格为whitegrid，背景更简洁

    # 绘制直方图
    plt.hist(original_crit, bins=30, density=True, alpha=0.5, color='blue', label='Human', edgecolor='none')
    plt.hist(rewritten_crit, bins=30, density=True, alpha=0.5, color='red', label='Machine', edgecolor='none')

    # 核密度估计
    kde_original = gaussian_kde(original_crit, bw_method=bandwidth)
    kde_sampled = gaussian_kde(rewritten_crit, bw_method=bandwidth)

    xmin, xmax = min(min(original_crit), min(rewritten_crit)), max(max(original_crit), max(rewritten_crit))
    x = np.linspace(xmin, xmax, 1000)
    plt.plot(x, kde_original(x), 'b-', label='Human KDE')
    plt.plot(x, kde_sampled(x), 'r-', label='Machine KDE')

    plt.title(f'Conditional Probability Curvature Distribution\nN={len(original_crit)} samples | Bandwidth={bandwidth}')
    plt.xlabel('Curvature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)  # 移除网格线
    plt.show()

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def experiment(args):
    original_crit = []
    rewritten_crit = []
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["rewritten"])
    # evaluate criterion
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    start_time = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        rewritten_text = data["rewritten"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit.append(criterion_fn(logits_ref, logits_score, labels))
        # rewritten text
        tokenized = scoring_tokenizer(rewritten_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(rewritten_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            rewritten_crit.append(criterion_fn(logits_ref, logits_score, labels))
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit[-1],
                        "rewritten": rewritten_text,
                        "rewritten_crit": rewritten_crit[-1]})
    print(f"Total time: {time.time() - start_time:.4f}s")
    # compute prediction scores for real/rewritten passages
    predictions = {'real': original_crit,
                   'samples': rewritten_crit}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # plot distributions
    plot_distributions(predictions['real'], predictions['samples'], bandwidth=0.3)
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default="./output", help="Path prefix for output results")
parser.add_argument('--dataset', type=str, default="data", help="Short name identifier of dataset")
parser.add_argument('--dataset_file', type=str, default="./data", help="Root directory storing dataset json files")
parser.add_argument('--reference_model_name', type=str, default="gpt", help="Name or local path of reference LLM")
parser.add_argument('--scoring_model_name', type=str, default="gpt", help="Name or local path of scoring LLM")
parser.add_argument('--discrepancy_analytic', action='store_true', help="Enable discrepancy analytic calculation mode")
parser.add_argument('--seed', type=int, default=0, help="Random seed to guarantee experiment reproducibility")
parser.add_argument('--device', type=str, default="cuda", help="Device for model inference, cuda or cpu")
parser.add_argument('--cache_dir', type=str, default="./local_model", help="Cache folder for local pretrained models")
args = parser.parse_args()
    experiment(args)


