import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from sklearn.metrics import roc_curve


class EvalDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 确保数据格式正确
        assert "original" in self.data and "rewritten" in self.data, \
            "数据集需要包含original和rewritten字段"
        assert len(self.data["original"]) == len(self.data["rewritten"]), \
            "原始文本和生成文本数量不一致"

    def __len__(self):
        return len(self.data["original"])

    def __getitem__(self, idx):
        return {
            "original": self.data["original"][idx],
            "rewritten": self.data["rewritten"][idx]
        }


def calculate_lrp_diff(model_scoring, tokenizer_scoring,
                       model_ref, tokenizer_ref, text, device):
    """计算两个模型的LRP差值"""

    def _get_lrp(model, tokenizer):
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            if inputs.input_ids.shape[1] == 0:
                return 0.0  # 处理空文本

            labels = inputs.input_ids[:, 1:]
            logits = model(**inputs).logits[:, :-1]

            # 计算对数概率
            log_probs = torch.log_softmax(logits, dim=-1)
            log_likelihood = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()

            # 计算对数排名
            sorted_ids = torch.argsort(logits, dim=-1, descending=True)
            ranks = (sorted_ids == labels.unsqueeze(-1)).nonzero()[:, -1] + 1
            log_rank = torch.log(ranks.float()).mean()

            return (log_likelihood * log_rank).item()

    try:
        lrp_scoring = _get_lrp(model_scoring, tokenizer_scoring)
        lrp_ref = _get_lrp(model_ref, tokenizer_ref)
        return lrp_scoring - lrp_ref
    except Exception as e:
        print(f"计算LRP时发生错误: {str(e)}")
        return 0.0


def evaluate_models(args):
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ======================================
    # 1. 加载基础模型（Base Model）
    # ======================================
    print(f"加载基础模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map='auto',
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ======================================
    # 2. 加载 Adapter 模型
    # ======================================
    print(f"加载评分模型（Adapter）: {args.scoring_model}")
    peft_config = PeftConfig.from_pretrained(args.scoring_model)
    model_scoring = PeftModel.from_pretrained(
        base_model,
        args.scoring_model,
        device_map='auto',
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    tokenizer_scoring = base_tokenizer

    # ======================================
    # 3. 加载参考模型
    # ======================================
    print(f"加载参考模型: {args.reference_model}")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.reference_model,
        device_map='auto',
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer_ref = AutoTokenizer.from_pretrained(args.reference_model)

    # 加载数据集
    print(f"加载数据集: {args.data_path}")
    dataset = EvalDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 执行评估
    results = {"original": [], "rewritten": []}

    for batch in tqdm(dataloader, desc="评估进度"):
        for texts, key in [(batch["original"], "original"), (batch["rewritten"], "rewritten")]:
            for text in texts:
                if not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                diff = calculate_lrp_diff(
                    model_scoring, tokenizer_scoring,
                    model_ref, tokenizer_ref,
                    text, device
                )
                results[key].append(diff)

    # 计算评估指标
    y_true = np.array([0] * len(results["original"]) + [1] * len(results["rewritten"]))
    y_scores = np.array(results["original"] + results["rewritten"])

    # 自动确定分数方向
    if np.mean(y_scores[y_true == 1]) < np.mean(y_scores[y_true == 0]):
        y_scores = -y_scores

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # 指标结果
    result = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "original_mean": float(np.mean(results["original"])),
        "original_std": float(np.std(results["original"])),
        "generated_mean": float(np.mean(results["rewritten"])),
        "generated_std": float(np.std(results["rewritten"])),
    }

    # 绘图保存
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, "curves.png"))
    plt.close()

    # 保存json指标
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print("\n评估结果:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"原始文本平均LRP差值: {result['original_mean']:.4f} ± {result['original_std']:.4f}")
    print(f"生成文本平均LRP差值: {result['generated_mean']:.4f} ± {result['generated_std']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文本检测模型评估程序")
    # 参考模型、基础模型默认放在项目本地 local_model 文件夹
    parser.add_argument('--reference_model', type=str,
                        default="./local_model/gpt")
    parser.add_argument('--base_model', type=str,
                        default="./local_model/gpt")
    # LoRA微调后的评分模型
    parser.add_argument('--scoring_model', type=str,
                        default="./aspo/ckpt/scoring_model")
    # 验证数据集路径
    parser.add_argument('--data_path', type=str,
                        default="./dataset")
    # 输出图表与指标文件夹
    parser.add_argument('--output_dir', type=str,
                        default="./aspo/output11")
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_models(args)