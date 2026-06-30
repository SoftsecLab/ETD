from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
# 原代码
# def from_pretrained(cls, model_name, kwargs, cache_dir):
#     if "/" in model_name:
#         local_path = os.path.join(cache_dir, model_name.split("/")[1])
#     else:
#         local_path = os.path.join(cache_dir, model_name)
#     if os.path.exists(local_path):
#         return cls.from_pretrained(local_path, **kwargs)
#     return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir,device_map='auto')
def from_pretrained(cls, model_name, kwargs, cache_dir):
    # 检查模型名称是否包含路径分隔符，如果是，则直接使用该路径
    if os.path.isabs(model_name):
        local_path = model_name
    else:
        # 否则，将模型名称与缓存目录结合，形成完整的本地路径
        local_path = os.path.join(cache_dir, model_name)

    # 检查本地路径是否存在
    if os.path.exists(local_path) and os.path.isdir(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    else:
        # 如果本地路径不存在，尝试从远程仓库加载模型
        return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')
# predefined models
model_fullnames = {  'gpt2': 'gpt2',
                     'gpt2-xl': 'gpt2-xl',
                     'opt-2.7b': 'facebook/opt-2.7b',
                     'gpt-neo-2.7B': 'EleutherAI/gpt-neo-2.7B',
                     'gpt-j-6B': 'EleutherAI/gpt-j-6B',
                     'gpt-neox-20b': 'EleutherAI/gpt-neox-20b',
                     'mgpt': 'sberbank-ai/mGPT',
                     'pubmedgpt': 'stanford-crfm/pubmedgpt',
                     'mt5-xl': 'google/mt5-xl',
                     'llama-13b': 'huggyllama/llama-13b',
                     'llama2-13b': 'TheBloke/Llama-2-13B-fp16',
                     'bloom-7b1': 'bigscience/bloom-7b1',
                     'opt-13b': 'facebook/opt-13b',
                     'uergpt2-distil-chinese-cluecorpussmall': 'uergpt2-distil-chinese-cluecorpussmall',
                     't5-small': 'google/t5-small',
                     'llama-3.2-1b': 'D:\\models\\Llama-3.2-1B',
                     }
float16_models = ['gpt-j-6B', 'gpt-neox-20b', 'llama-13b', 'llama2-13b', 'bloom-7b1', 'opt-13b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name
# 原代码
# def load_model(model_name, device, cache_dir):
#     model_fullname = get_model_fullname(model_name)
#     print(f'Loading model {model_fullname}...')
#     model_kwargs = {}
#     if model_name in float16_models:
#         model_kwargs.update(dict(torch_dtype=torch.float16))
#     if 'gpt-j' in model_name:
#         model_kwargs.update(dict(revision='float16'))
#     model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
#     print('Moving model to GPU...', end='', flush=True)
#     start = time.time()
#     model.to(device)
#     print(f'DONE ({time.time() - start:.2f}s)')
#     return model
def load_model(model_name, device, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {}
    if model_name in float16_models:
        model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in model_fullname:
        model_kwargs.update(dict(revision='float16'))
    model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
    print('Moving model to GPU...', end='', flush=True)
    start = time.time()
    model.to(device)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model
# 原代码
# def load_tokenizer(model_name, for_dataset, cache_dir):
#     model_fullname = get_model_fullname(model_name)
#     optional_tok_kwargs = {}
#     if "facebook/opt-" in model_fullname:
#         print("Using non-fast tokenizer for OPT")
#         optional_tok_kwargs['fast'] = False
#     if for_dataset in ['pubmed']:
#         optional_tok_kwargs['padding_side'] = 'left'
#     else:
#         optional_tok_kwargs['padding_side'] = 'right'
#     base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
#     if base_tokenizer.pad_token_id is None:
#         base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
#         if '13b' in model_fullname:
#             base_tokenizer.pad_token_id = 0
#     return base_tokenizer
def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    optional_tok_kwargs = {}
    if "facebook/opt-" in model_fullname:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if for_dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    else:
        optional_tok_kwargs['padding_side'] = 'right'
    base_tokenizer = from_pretrained(AutoTokenizer, model_fullname, optional_tok_kwargs, cache_dir=cache_dir)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        if '13b' in model_fullname:
            base_tokenizer.pad_token_id = 0
    return base_tokenizer

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, default="uergpt2-distil-chinese-cluecorpussmall")
#     parser.add_argument('--cache_dir', type=str, default="../cache")
#     args = parser.parse_args()
#
#     load_tokenizer(args.model_name, 'xsum', args.cache_dir)
#     load_model(args.model_name, 'cpu', args.cache_dir)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="D:\\python_study\pythonProject2\\models\\uergpt2-distil-chinese-cluecorpussmall")
    parser.add_argument('--cache_dir', type=str, default="D:\python_study\pythonProject2\models")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name, 'xsum', args.cache_dir)
    model = load_model(args.model_name, args.device, args.cache_dir)