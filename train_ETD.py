import torch
import argparse
import numpy as np
import random
import os
import json
from torch.utils.data import Dataset, Subset
from etd import ComputeScore
from trainer import run
from tqdm import tqdm

# --- 数据集类嵌入 ---
class CustomDataset_rewrite(Dataset):
    def __init__(self, data_json_dir):
        with open(data_json_dir, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        self.data = self.process_data(data_json)

    def __len__(self):
        return len(self.data['original'])

    def __getitem__(self, index):
        return self.data['original'][index], self.data['rewritten'][index]

    def process_data(self, data_json):
        return {'original': data_json['original'], 'rewritten': data_json['rewritten']}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 核心参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--gama', type=float, default=1.0)
    parser.add_argument('-a', type=int, default=4, help="accumulation steps")
    parser.add_argument('--task_name', type=str, default="ai_detection_etd")
    parser.add_argument('--epochs', type=int, default=2, help="finetuning epochs")
    parser.add_argument('--val_freq', type=int, default=1, help="frequency of eval and saving model")
    
    # 路径配置 (已替换为通用相对路径)
    parser.add_argument('--eval_dataset', type=str, default="./data/eval_data.json", help="Path to evaluation dataset")
    parser.add_argument('--train_dataset', type=str, default='./data/train_data.json', help="Path to training dataset")
    parser.add_argument('--cache_dir', type=str, default="./models/cache", help="Directory for model cache")
    parser.add_argument('--paraphraser_model', type=str, default="./models/t5-base", help="Path to T5 paraphraser model")
    parser.add_argument('--base_model', type=str, default="./models/gpt",help="Path to scoring model")
    
    # 其他配置
    parser.add_argument('--datanum', type=int, default=500, help="num of training data")
    parser.add_argument('--ETDtrained', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--output_file', type=str, default="./results/output")
    
    args = parser.parse_args()

    # 初始化环境
    set_seed(args.seed)
    ETDtrained = True if args.ETDtrained == "True" else False
    
    # 初始化 ETD 模型
    model = ComputeScore(
        scoring_model_name=args.base_model, 
        reference_model_name=args.base_model, 
        paraphraser_model=args.paraphraser_model,
        ETDtrained=ETDtrained, 
        gama=args.gama, 
        ETD_beta=args.beta, 
        cache_dir=args.cache_dir
    )
    
    if args.from_pretrained:
        model.from_pretrained(args.from_pretrained)
        
    train_data = CustomDataset_rewrite(data_json_dir=args.train_dataset) 
    val_data = CustomDataset_rewrite(data_json_dir=args.eval_dataset)
    
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)
        
    subset_indices = torch.randperm(len(train_data))[:args.datanum]
    train_subset = Subset(train_data, subset_indices)
    
    run(model, [train_subset, val_data], DEVICE='cuda', 
        ckpt_dir=f"./ckpt/{args.task_name}_etd", args=args)