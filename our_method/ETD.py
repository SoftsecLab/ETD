import torch
from pyarrow.compute import top_k_unstable
from torch import nn
import sys
import json
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM,PeftModel
import os
import torch.nn.functional as F
from torch.optim import AdamW

from utils import calculate_ETD_loss
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig,AutoModelForMaskedLM
import time
from transformers import (
    T5ForConditionalGeneration,
    RobertaForSequenceClassification,
    T5Tokenizer,
    RobertaTokenizer
)
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def from_pretrained(cls, model_name, kwargs, cache_dir):
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

model_fullnames = {  
    'gpt-neo-2.7B': 'gpt-neo-2.7B',
}
float16_models = ['gpt-j-6B']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

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


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    if logits_ref.size(-1) != logits_score.size(-1):
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

    return discrepancy, log_likelihood.mean(dim=-1)


class ComputeScore(nn.Module):
    def __init__(self, scoring_model_name, reference_model_name, paraphraser_model, ETDtrained=True, gama=0.5, ETD_beta=0.5, dataset='xsum', device='cuda', cache_dir='./models', ppo_epsilon=0.2, entropy_coef=0.01, **kwargs):
        super().__init__()
        self.device = device
        self.reference_model_name = get_model_fullname(reference_model_name)
        self.scoring_model_name = get_model_fullname(scoring_model_name)
        self.beta = ETD_beta
        self.gama = gama
        self.paraphraser = T5ForConditionalGeneration.from_pretrained(paraphraser_model, cache_dir=cache_dir).to(device)
        self.paraphraser_tokenizer = T5Tokenizer.from_pretrained(paraphraser_model)
        self.paraphraser_optim = AdamW(self.paraphraser.parameters(), lr=5e-4)
        self.ppo_epsilon = ppo_epsilon  
        self.entropy_coef = entropy_coef  
        self.reward_beta = nn.Parameter(torch.tensor(ETD_beta))
        self.reward_ema_alpha = 0.95  

        def load_model(model_name, device, cache_dir, ETDtrained=True):
            model_fullname = get_model_fullname(model_name)
            print(f'Loading model {model_fullname}...')
            model_kwargs = {}
            if model_name in float16_models:
                model_kwargs.update(dict(torch_dtype=torch.float16))
            if 'gpt-j' in model_name:
                model_kwargs.update(dict(revision='float16'))
            if ETDtrained:
                model = from_pretrained(AutoModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            else: 
                model = from_pretrained(AutoPeftModelForCausalLM, model_fullname, model_kwargs, cache_dir)
            print('Moving model to GPU...', end='', flush=True)
            start = time.time()
            model.to(device)
            print(f'DONE ({time.time() - start:.2f}s)')
            return model
        
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, dataset, cache_dir)
        self.target_token_id = self.scoring_tokenizer.convert_tokens_to_ids("positive")  
        if self.target_token_id is None:  
            self.target_token_id = self.scoring_tokenizer.eos_token_id  
        
        scoring_model = load_model(scoring_model_name, device, cache_dir, ETDtrained)

        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
            fan_in_fan_out = True
        )

        reference_model = load_model(reference_model_name, device, cache_dir, ETDtrained)
        
        if ETDtrained: 
            self.scoring_model = get_peft_model(scoring_model, self.peft_config)
            self.reference_model = get_peft_model(reference_model, self.peft_config)
        else: 
            self.scoring_model = scoring_model
            self.reference_model = reference_model
            
        self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)
             
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.forward = self.forward_ETD

        self.scoring_model.print_trainable_parameters()

    def generate_perturbation(self, texts, current_step=0, total_steps=1000):
        valid_texts = [text if isinstance(text, str) and len(text) > 0 else "[PAD]" for text in texts]
        if isinstance(valid_texts[0], list):
                 valid_texts = [t[0] for t in valid_texts]
        perturbed_texts = []
        for text in valid_texts:
            encoded = self.paraphraser_tokenizer(text, return_tensors="pt", truncation=True, max_length=200).to(self.device)
            input_ids = encoded.input_ids[0]
            seq_len = len(input_ids)

            with torch.no_grad():
                outputs = self.paraphraser(**encoded, labels=encoded.input_ids)
                logits = outputs.logits[:, :-1]
                labels = encoded.input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]

            special_ids = set(self.paraphraser_tokenizer.all_special_ids)
            valid_indices = [
                i for i in range(1, seq_len - 1)
                if input_ids[i].item() not in special_ids
            ]

            if len(valid_indices) < 4:
                perturbed_texts.append(text)
                continue

            perturbation_ratio = random.uniform(0.5, 0.10)
            num_mask = max(1, int(len(valid_indices) * perturbation_ratio))
            mask_indices = random.sample(valid_indices, num_mask)

            masked_ids = input_ids.clone()
            mask_token_id = self.paraphraser_tokenizer.mask_token_id or self.paraphraser_tokenizer.convert_tokens_to_ids("<mask>")
            for idx in mask_indices:
                masked_ids[idx] = mask_token_id

            masked_text = self.paraphraser_tokenizer.decode(masked_ids, skip_special_tokens=True)

            inputs = self.paraphraser_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=200).to(self.device)
            with torch.no_grad():
                generated_ids = self.paraphraser.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    do_sample=True,
                    top_k=30,
                    top_p=0.95,
                    temperature=1.0,
                    max_length=200,
                    pad_token_id=self.paraphraser_tokenizer.eos_token_id
                )

            generated_ids = generated_ids[0]
            perturbed_ids = input_ids.clone()
            min_len = min(len(generated_ids), len(input_ids))
            for idx in mask_indices:
                if idx < min_len:
                    perturbed_ids[idx] = generated_ids[idx]

            perturbed_text = self.paraphraser_tokenizer.decode(perturbed_ids, skip_special_tokens=True)
            perturbed_texts.append(perturbed_text)

        return perturbed_texts

    def update_paraphraser(self, original_texts, perturbed_texts, Δ_p, Δ_d):
        Δ_p = torch.tensor(Δ_p, dtype=torch.float32, device=self.device, requires_grad=True)
        Δ_d = torch.tensor(Δ_d, dtype=torch.float32, device=self.device, requires_grad=True)
        
        with torch.no_grad():
            if not hasattr(self, 'reward_mean'):
                self.reward_mean = torch.tensor(0.0, device=self.device)
                self.reward_std = torch.tensor(1e-6, device=self.device)

            current_rewards = self.calculate_reward(Δ_p, Δ_d)
            batch_mean = current_rewards.mean()
            batch_var = current_rewards.var(unbiased=False)  

            self.reward_mean = self.reward_ema_alpha * self.reward_mean + \
                               (1 - self.reward_ema_alpha) * batch_mean

            delta_mean = batch_mean - self.reward_mean
            ema_var = self.reward_ema_alpha * (self.reward_std ** 2 + delta_mean ** 2) + \
                      (1 - self.reward_ema_alpha) * batch_var
            self.reward_std = torch.sqrt(ema_var).clamp_min(1e-6)
            
        inputs = self.paraphraser_tokenizer(
            original_texts, padding=True, truncation=True, max_length=200, return_tensors="pt"
        ).to(self.device)
        
        labels = self.paraphraser_tokenizer(
            perturbed_texts, padding=True, truncation=True, max_length=200, return_tensors="pt"
        ).to(self.device)
        
        decoder_input_ids = self.paraphraser._shift_right(labels.input_ids)
        
        with torch.no_grad():
            old_outputs = self.paraphraser(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                decoder_input_ids=decoder_input_ids, return_dict=True
            )
            old_logits = old_outputs.logits
            
        new_outputs = self.paraphraser(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            decoder_input_ids=decoder_input_ids, return_dict=True
        )
        new_logits = new_outputs.logits
        
        probs = torch.softmax(new_logits, dim=-1)
        old_probs = torch.softmax(old_logits, dim=-1)

        target_ids = labels.input_ids.unsqueeze(-1)
        selected_probs = probs.gather(-1, target_ids).squeeze(-1)
        old_selected_probs = old_probs.gather(-1, target_ids).squeeze(-1)
        ratio = selected_probs / (old_selected_probs + 1e-8)
        
        advantages = (current_rewards - self.reward_mean) / (self.reward_std + 1e-8)
        advantages = advantages.detach().unsqueeze(1).expand(-1, ratio.size(1))
        
        clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        valid_mask = (labels.input_ids != self.paraphraser_tokenizer.pad_token_id).float()
        policy_loss = (policy_loss * valid_mask).sum() / valid_mask.sum()
        
        entropy = -torch.mean(probs * torch.log(probs + 1e-8))
        entropy_loss = -self.entropy_coef * entropy
        total_loss = policy_loss + entropy_loss
        
        self.paraphraser_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.paraphraser.parameters(), 1.0)
        self.paraphraser_optim.step()
        return total_loss.item()

    def calculate_reward(self, Δ_p, Δ_d):
        
        reward_diff = Δ_p - Δ_d
        sigmoid_term = torch.sigmoid(self.reward_beta * reward_diff)
        return 1.0 - sigmoid_term  

    def print_gradient_requirement(self):
        for name, param in self.named_parameters():
            gradient_requirement = 'Requires Grad' if param.requires_grad else 'Does not require grad'
            color_code = '\033[92m' if param.requires_grad else '\033[91m'  
            reset_color = '\033[0m'  
            print(f"{name}: {color_code}{gradient_requirement}{reset_color}")

    def register_no_grad(self, module_names):
        for name, param in self.named_parameters():
            for selected_module in module_names:
                if selected_module in name:
                    param.requires_grad = False

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.scoring_model.save_pretrained(save_directory)
        self.scoring_tokenizer.save_pretrained(save_directory)
        
    def from_pretrained(self, load_directory):
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")
        self.load_state_dict(torch.load(os.path.join(load_directory, "model.bin"), map_location=self.device))

    def get_ETD_input(self, tokenized=None ,text=[""], labels=[""], training_module=False):
        if training_module:
            logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
            logits_ref = logits_score
            if self.reference_model_name != self.scoring_model_name:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False, add_special_tokens=True, return_attention_mask=True).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(tokenized.input_ids).logits[:,:-1,:]
            crit, ETD_input  = self.criterion_fn(logits_ref, logits_score, labels)
        else:
            with torch.no_grad(): 
                if self.reference_model_name != self.scoring_model_name:
                    tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False ,add_special_tokens=True, return_attention_mask=True).to(self.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_score = self.reference_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:] 
                logits_ref = logits_score
                crit, ETD_input = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, ETD_input, logits_score

    def forward_ETD(self, text):
        original_text = text[0]
        sampled_text = text[1]
        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] 
        ref_original_crit, ref_disprefered_logprob, ref_original_logits_score = self.get_ETD_input(tokenized,original_text,labels)
        train_original_crit, train_disprefered_logprob, train_original_logits_score = self.get_ETD_input(tokenized,original_text,labels,training_module=True)
        
        tokenized = self.scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        ref_sampled_crit, ref_prefered_logprob, ref_sampled_logits_score = self.get_ETD_input(tokenized,sampled_text,labels)
        train_sampled_crit, train_prefered_logprob, train_sampled_logits_score = self.get_ETD_input(tokenized,sampled_text,labels,training_module=True)


        ETDloss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_ETD_loss(
            train_prefered_logprob, train_disprefered_logprob, ref_prefered_logprob, ref_disprefered_logprob, beta=self.beta, gama=self.gama
        )
        output = dict(crit=[ref_original_crit, train_original_crit, ref_sampled_crit, train_sampled_crit], loss=ETDloss, r_l=[prefered_relative_logprob, disprefered_relative_logprob])
        return output