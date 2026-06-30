import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import os

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()


def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1
    ranks = torch.log(ranks)
    return ranks.mean().item()


def calculate_lrp(model, tokenizer, text, device):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
        labels = inputs.input_ids[:, 1:]
        logits = model(**inputs).logits[:, :-1]

        likelihood = get_likelihood(logits, labels)
        logrank = get_logrank(logits, labels)
        return likelihood * logrank


def main(args):
    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型和分词器
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    model.eval()

    # 加载数据集
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)

    # 计算两组 LRP 指标
    results = {"human": [], "ai": []}
    log_likelihoods = {"human": [], "ai": []}

    # 处理人类文本
    for text in tqdm(dataset["original"], desc="Processing Human Texts"):
        try:
            lrr = calculate_lrp(model, tokenizer, text, args.device)
            results["human"].append(lrr)
            
            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
            logits = model(**inputs).logits[:, :-1]
            labels = inputs.input_ids[:, 1:]
            log_likelihood = get_likelihood(logits, labels)
            log_likelihoods["human"].append(log_likelihood)

        except Exception as e:
            print(f"Error processing human text: {str(e)}")
            continue
            
    # 处理AI文本
    for text in tqdm(dataset["rewritten"], desc="Processing AI Texts"):
        try:
            lrr = calculate_lrp(model, tokenizer, text, args.device)
            results["ai"].append(lrr)
            
            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
            logits = model(**inputs).logits[:, :-1]
            labels = inputs.input_ids[:, 1:]
            log_likelihood = get_likelihood(logits, labels)
            log_likelihoods["ai"].append(log_likelihood)
            
        except Exception as e:
            print(f"Error processing AI text: {str(e)}")
            continue

    # 保存对数概率值
    with open(os.path.join(args.save_dir, "log_likelihoods.json"), "w", encoding="utf-8") as f:
        json.dump(log_likelihoods, f, ensure_ascii=False, indent=4)

    # ==============================
    # 可视化结果与计算 AUROC / PR AUC
    # ==============================
    
    # 1. 绘制 LRP 分数分布直方图
    plt.figure(figsize=(10, 6))
    bins = np.linspace(
        min(results["human"] + results["ai"]),
        max(results["human"] + results["ai"]),
        30
    )
    plt.hist(results["human"], bins=bins, alpha=0.5, label="Human Texts", color="blue")
    plt.hist(results["ai"], bins=bins, alpha=0.5, label="AI Texts", color="red")

    # 添加均值辅助线
    plt.axvline(np.mean(results["human"]), color="blue", linestyle="dashed", linewidth=1)
    plt.axvline(np.mean(results["ai"]), color="red", linestyle="dashed", linewidth=1)

    plt.title("LRP Score Distribution")
    plt.xlabel("LRP Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, "lrp_distribution.png"), dpi=300)
    plt.close()

    # 2. 准备分类标签和分数
    y_true = np.array([0] * len(results["human"]) + [1] * len(results["ai"]))
    y_scores = np.array(results["human"] + results["ai"])
    
    # 自动确定分数方向，确保 AI 分数更高用于 ROC 计算
    if np.mean(results["ai"]) < np.mean(results["human"]):
        y_scores = -y_scores  

    # 3. 计算指标
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prauc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # 4. 绘制独立 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 5. 绘制双子图 (ROC & PR)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auroc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {prauc:.2f})')
    ax2.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "roc_pr_curve.png"), dpi=300)
    plt.close()

    # 输出两个指标
    print(f"\nAUROC: {auroc:.4f}")
    print(f"PR AUC: {prauc:.4f}")
    print(f"All plots and data saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LRP scores and evaluate detection performance.")
    
    # 替换原本的硬编码路径为命令行参数
    parser.add_argument('--model_path', type=str, required=True, help="Path to the local model or huggingface model name.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the JSON dataset file.")
    parser.add_argument('--save_dir', type=str, default="./results", help="Directory to save the outputs and plots.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    
    args = parser.parse_args()
    main(args)