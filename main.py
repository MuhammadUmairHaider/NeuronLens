#!/usr/bin/env python
import argparse
import os
import torch
import random
import numpy as np
import warnings
from datasets import load_dataset, concatenate_datasets, ClassLabel
from collections import Counter

# Specific evaluations depending on the model and dataset
from utils import (
    evaluate_gpt2_classification,
    evaluate_llma_classification_batch_emotions,
    evaluate_llma_classification_ag_news,
    evaluate_llma_classification_batch_db14,
    evaluate_llma_language_modeling,
    mask_range_llma,
    mask_range_gpt,
    compute_masks,
    reset_llma,
    reset_gpt,
    extract_avg,
    mask_gpt2
)

from prettytable import PrettyTable

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def balance_by_samples(ds, label_col="label", samples_per_class=None, seed=SEED, do_balance=False):
    labels = ds[label_col]
    counts = Counter(labels)
    if do_balance:
        min_n = min(counts.values())
        if samples_per_class:
            min_n = min(min_n, samples_per_class)
    else:
        # Not strictly balancing, just capping
        min_n = samples_per_class if samples_per_class else max(counts.values())

    rng = np.random.default_rng(seed)
    shards = []
    for cls in sorted(counts.keys()):
        idx = [i for i, y in enumerate(labels) if y == cls]
        sz = min(len(idx), min_n)
        pick = rng.choice(idx, size=sz, replace=False)
        shards.append(ds.select(sorted(pick.tolist())))
    balanced = concatenate_datasets(shards).shuffle(seed=seed)
    return balanced

def get_llama_eval_fn(dataset_name):
    if dataset_name == 'emotions':
        return evaluate_llma_classification_batch_emotions
    elif dataset_name == 'ag_news':
        return evaluate_llma_classification_ag_news
    elif dataset_name == 'db14':
        return evaluate_llma_classification_batch_db14
    return evaluate_llma_classification_batch_emotions

def load_llama_model(layer):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from models.lama import LlamaForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Llama-3.2-3B",
                    pad_token_id=tokenizer.eos_token_id
                )
    base_model.config.m_layer = layer
    base_model.config.mask_layer_type = "hard"
    
    model = LlamaForCausalLM(base_model.config)
    model.load_state_dict(base_model.state_dict())
    del base_model
    
    model.to("cuda")
    return model, tokenizer

def load_gpt2_model(layer, tokenizer_len, dataset_name):
    from transformers import GPT2LMHeadModel as gt
    from models.gpt2 import GPT2LMHeadModel
    
    model1 = gt.from_pretrained('gpt2')
    model1.resize_token_embeddings(tokenizer_len)
    model1.config.m_layer = layer
    
    weights_path = os.path.join("model_weights", "fancyzhx/ag_news", "weights.pth") # original default
    model = GPT2LMHeadModel(model1.config)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    else:
        print(f"Warning: {weights_path} not found. Proceeding with uninitialized custom weights.")
    model.to("cuda")
    return model

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for GPT2 and Llama")
    parser.add_argument("--model", type=str, required=True, choices=['gpt2', 'llama'], help="Model to use")
    parser.add_argument("--dataset", type=str, required=True, choices=['ag_news', 'emotions', 'db14'], help="Dataset to use")
    parser.add_argument("--percentage", type=float, default=0.3, help="Percentage of neurons to mask (e.g. 0.3 for 30%)")
    parser.add_argument("--tao", type=float, default=2.0, help="Tao value for range masking")
    parser.add_argument("--layer", type=int, default=27, help="Intermediate layer to mask")
    parser.add_argument("--samples_per_class", type=int, default=None, help="Number of samples to cap per class")
    parser.add_argument("--balance", action="store_true", help="Force dataset class balance based on lowest class count")
    
    args = parser.parse_args()
    
    set_seed(SEED)
    torch.autograd.set_detect_anomaly(True)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")
    
    print(f"Running config: Model={args.model}, Dataset={args.dataset}, Layer={args.layer}")
    
    if args.model == 'llama':
        dataset_files = {
            'emotions': "llama_correct_datasets/correct_predictions_emotions.json",
            'ag_news': "llama_correct_datasets/correct_predictions_ag_news.json",
            'db14': "llama_correct_datasets/correct_predictions_DB_14.json"
        }
        num_classes_map = {'emotions': 6, 'ag_news': 4, 'db14': 14}
        num_classes = num_classes_map[args.dataset]
        
        correct_ds = load_dataset("json", data_files=dataset_files[args.dataset])["train"]
        correct_ds = balance_by_samples(correct_ds, samples_per_class=args.samples_per_class, do_balance=args.balance)
        
        if not isinstance(correct_ds.features["label"], ClassLabel):
            unique = list(range(num_classes))
            label_feature = ClassLabel(num_classes=num_classes, names=[str(u) for u in unique])
            correct_ds = correct_ds.cast_column("label", label_feature)
            
        splits = correct_ds.train_test_split(test_size=0.20, seed=SEED, stratify_by_column="label")
        train_ds = splits["train"]
        test_ds = splits["test"]
        
        wiki_sample = load_dataset("wikipedia", "20220301.en", split="train[:2000]")
        mmlu_val = load_dataset("cais/mmlu", 'all', split="dev[:200]") if False else None # Optional auxiliary

        model, tokenizer = load_llama_model(args.layer)
        avg_token = extract_avg(model, wiki_sample, tokenizer, max_samples=200)
        
        eval_fn = get_llama_eval_fn(args.dataset)
        
        results_table = PrettyTable()
        results_table.field_names = [
            "Class", "State", "Tgt Acc", "Tgt Conf", "Comp Acc", "Comp Conf", "PPL"
        ]
        
        for cls in range(num_classes):
            print(f"\n=== Class {cls} ===")
            model = reset_llma(model)
            
            ds_pos = test_ds.filter(lambda x: x["label"] == cls)
            ds_comp = test_ds.filter(lambda x: x["label"] != cls)
            ds_rec = train_ds.filter(lambda x: x["label"] == cls)
            
            print("Evaluating Base...")
            b_tgt_acc, b_tgt_conf, _ = eval_fn(model, ds_pos, tokenizer)
            b_comp_acc, b_comp_conf, _ = eval_fn(model, ds_comp, tokenizer)
            b_ppl = evaluate_llma_language_modeling(model, wiki_sample, tokenizer, max_samples=200)
            results_table.add_row([f"Class {cls}", "Base", b_tgt_acc, b_tgt_conf, b_comp_acc, b_comp_conf, round(b_ppl, 4)])
            
            _, _, fc_vals = eval_fn(model, ds_rec, tokenizer)
            mask_max, *_ = compute_masks(fc_vals, args.percentage)
            
            print("Applying Range Masking...")
            model_rng = mask_range_llma(reset_llma(model), mask_max, fc_vals, avg_token, args.tao)
            r_tgt_acc, r_tgt_conf = eval_fn(model_rng, ds_pos, tokenizer)[:2]
            r_comp_acc, r_comp_conf = eval_fn(model_rng, ds_comp, tokenizer)[:2]
            r_ppl = evaluate_llma_language_modeling(model_rng, wiki_sample, tokenizer, max_samples=200)
            results_table.add_row(["", "Range Mask", r_tgt_acc, r_tgt_conf, r_comp_acc, r_comp_conf, round(r_ppl, 4)])
            
            print("Applying Full Masking...")
            model_max_mask = mask_range_llma(reset_llma(model), mask_max, fc_vals, avg_token, torch.inf)
            m_tgt_acc, m_tgt_conf = eval_fn(model_max_mask, ds_pos, tokenizer)[:2]
            m_comp_acc, m_comp_conf = eval_fn(model_max_mask, ds_comp, tokenizer)[:2]
            m_ppl = evaluate_llma_language_modeling(model_max_mask, wiki_sample, tokenizer, max_samples=200)
            results_table.add_row(["", "Max Mask", m_tgt_acc, m_tgt_conf, m_comp_acc, m_comp_conf, round(m_ppl, 4)])
            
        print("\\n===== RESULTS =====")
        print(results_table)
        
    elif args.model == 'gpt2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>', 'eos_token': '<|eos|>'}
        tokenizer.add_special_tokens(special_tokens)
        
        ds_map = {
            'ag_news': 'fancyzhx/ag_news',
            'emotions': 'dair-ai/emotion',
            'db14': 'dbpedia_14'
        }
        num_classes_map = {'emotions': 6, 'ag_news': 4, 'db14': 14}
        dataset = load_dataset(ds_map[args.dataset])
        
        # Determine columns
        if args.dataset == 'ag_news':
            text_tag, lab = 'text', 'label'
        elif args.dataset == 'emotions':
            text_tag, lab = 'text', 'label'
        else: # db14
            text_tag, lab = 'content', 'label'
            
        new_tokens = []
        try:
            label2text = dataset['train'].features[lab].names
            for label_name in label2text:
                new_tokens.extend([f'{label_name}'])
        except Exception:
            pass # fallback if custom names don't exist
            
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            
        model = load_gpt2_model(args.layer, len(tokenizer), ds_map[args.dataset])
        
        # Tokenize logic (simplified representation for full pipeline)
        def format_data(examples):
            formatted_texts = []
            for text, label in zip(examples[text_tag], examples[lab]):
                tok_text = tokenizer.encode(text, max_length=400, truncation=True)
                dec_text = tokenizer.decode(tok_text)
                formatted_texts.append(f"Classify emotion: {dec_text}{tokenizer.sep_token}")
            return {'formatted_text': formatted_texts}
            
        def tokenize_and_prepare(examples):
            tokenized = tokenizer(
                examples['formatted_text'],
                padding='max_length',
                max_length=408,
                truncation=True,
                return_tensors="pt"
            )
            labels = tokenized['input_ids'].clone()
            labels[:] = -100
            sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
            sep_positions = (labels == sep_token_id).nonzero(as_tuple=True)
            for batch_idx, sep_pos in zip(*sep_positions):
                if sep_pos + 1 < labels.size(1):
                    labels[batch_idx, sep_pos + 1] = tokenized['input_ids'][batch_idx, sep_pos + 1]
            labels[labels == tokenizer.pad_token_id] = -100
            return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': labels}
        
        dataset = dataset.filter(lambda x: x[lab] != -1)
        formatted_dataset = dataset.map(format_data, batched=True)
        tokenized_dataset = formatted_dataset.map(tokenize_and_prepare, batched=True)
        
        tokenized_dataset['train'] = balance_by_samples(tokenized_dataset['train'], label_col=lab, samples_per_class=args.samples_per_class, do_balance=args.balance)
        tokenized_dataset['test'] = balance_by_samples(tokenized_dataset['test'], label_col=lab, samples_per_class=args.samples_per_class, do_balance=args.balance)

        all_fc_vals = []
        num_classes = num_classes_map[args.dataset]
        results_table = PrettyTable()
        results_table.field_names = [
            "Class", "State", "Tgt Acc", "Tgt Conf", "Comp Acc", "Comp Conf"
        ]
        
        for j in range(num_classes):
            print(f"\n=== Class {j} ===")
            dataset_recording = tokenized_dataset['train'].filter(lambda x: x[lab] in [j])
            dataset_test = tokenized_dataset['test'].filter(lambda x: x[lab] in [j])
            dataset_comp = tokenized_dataset['test'].filter(lambda x: x[lab] not in [j])
            
            print("Evaluating Base...")
            b_tgt_acc, b_tgt_conf = evaluate_gpt2_classification(lab, model, dataset_test, tokenizer)[:2]
            b_comp_acc, b_comp_conf = evaluate_gpt2_classification(lab, model, dataset_comp, tokenizer)[:2]
            results_table.add_row([f"Class {j}", "Base", b_tgt_acc, b_tgt_conf, b_comp_acc, b_comp_conf])
            
            fc_vals = evaluate_gpt2_classification(lab, model, dataset_recording, tokenizer)[2]
            mask_max, *_ = compute_masks(fc_vals, args.percentage)
            
            print("Applying Range Masking...")
            model = mask_range_gpt(reset_gpt(model), mask_max, fc_vals, args.tao, [])
            r_tgt_acc, r_tgt_conf = evaluate_gpt2_classification(lab, model, dataset_test, tokenizer)[:2]
            r_comp_acc, r_comp_conf = evaluate_gpt2_classification(lab, model, dataset_comp, tokenizer)[:2]
            results_table.add_row(["", "Range Mask", r_tgt_acc, r_tgt_conf, r_comp_acc, r_comp_conf])
            
            print("Applying Full Masking...")
            model = mask_range_gpt(reset_gpt(model), mask_max, fc_vals, float('inf'), [])
            m_tgt_acc, m_tgt_conf = evaluate_gpt2_classification(lab, model, dataset_test, tokenizer)[:2]
            m_comp_acc, m_comp_conf = evaluate_gpt2_classification(lab, model, dataset_comp, tokenizer)[:2]
            results_table.add_row(["", "Max Mask", m_tgt_acc, m_tgt_conf, m_comp_acc, m_comp_conf])
            
            model = reset_gpt(model)
            
        print("\\n===== RESULTS =====")
        print(results_table)

if __name__ == "__main__":
    main()
