import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import torch
from scipy import stats

def compute_masks(fc_vals, percent):
    fc_vals_array = np.array(fc_vals)
    mean_vals = np.mean(np.abs(fc_vals_array), axis=0)
    mean_vals_tensor = torch.from_numpy(mean_vals)
    mask_max = compute_max_mask(mean_vals_tensor, percent)
    return (mask_max, )

def compute_max_mask(values, percent):
    sorted_indices = torch.argsort(values, descending=True)
    mask_count = int(percent * len(values))
    mask = torch.ones_like(values)
    mask[sorted_indices[:mask_count]] = 0.0
    return mask

def mask_gpt2(model, mask):
    model.transformer.mask_m_layer = mask.to(device)
    return model
import nethook
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

def collate_fn(batch):
    return {'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]), 'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]), 'label': torch.stack([torch.tensor(item['label']) for item in batch])}

def manual_generate_v2(model, input_ids, attention_mask):
    with torch.no_grad():
        with nethook.TraceDict(model, ['transformer.mask_layer']) as ret:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            fc1_vals = [ret[layer_fc1_vals].output[:, -1, :].to('cpu') for layer_fc1_vals in ret]
        next_token_logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        confidences = torch.nn.functional.softmax(next_token_logits, dim=-1)
        fc1_vals = fc1_vals[0]
        return (next_tokens, confidences, fc1_vals)

def evaluate_gpt2_classification(lab, model, eval_dataset, tokenizer, batch_size=128):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    confidence = 0
    all_hidden = []
    correct = 0
    j = 0
    return_dataset = []
    dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for item in tqdm(dataloader, desc='Evaluating'):
        input_ids = torch.tensor(item['input_ids']).to(device)
        attention_mask = torch.tensor(item['attention_mask']).to(device)
        with torch.no_grad():
            generated_token, confidences, fc_vals = manual_generate_v2(model, input_ids, attention_mask)
        for true_label, predicted_token, conf, fc in zip(item[lab], generated_token, confidences, fc_vals):
            true_label = tokenizer.encode(eval_dataset.features[lab].int2str(true_label.item()), add_special_tokens=False, truncation=True, return_tensors='pt').squeeze()
            predicted_label = predicted_token
            confidence += conf[true_label].cpu().numpy().item()
            j += 1
            if predicted_label == true_label:
                correct += 1
                return_dataset.append(item)
            all_hidden.append(fc.numpy())
    if j == 0:
        return (0, 0, [])
    return (round(correct / j, 4), round(confidence / j, 4), all_hidden, return_dataset)

def mask_range_gpt(model, mask, fc_vals, tao, all_fc_vals):
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao * std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao * std[~mask]
    model.transformer.mask_layer.lower_bound = lower_bound.to(device)
    model.transformer.mask_layer.upper_bound = upper_bound.to(device)
    return model

def reset_gpt(model):
    model.transformer.mask_layer.lower_bound = torch.tensor(float('inf')).to(device)
    model.transformer.mask_layer.upper_bound = torch.tensor(float('-inf')).to(device)
    return model
import nethook
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm
import torch
import torch
from torch import nn
from typing import Tuple
from tqdm import tqdm
import torch, nethook

def extract_avg(model, eval_dataset, tokenizer, max_samples: int=500, max_length: int=512, trace_module: str='model.mask_layer'):
    """
    Returns
    -------
    dict with
        - "perplexity": float
        - "avg_fc"    : torch.Tensor  (hidden_size,)
    """
    device = next(model.parameters()).device
    model.eval()
    total_loss, total_tokens = (0.0, 0)
    fc_sum = None
    n_examples = 0
    pbar = tqdm(eval_dataset, total=max_samples, desc='LM / FC-all')
    for example in pbar:
        inputs = tokenizer(example['text'], return_tensors='pt', truncation=True, max_length=max_length).to(device)
        seq_len = inputs['input_ids'].size(1)
        last_pos = seq_len - 1
        with torch.no_grad(), nethook.TraceDict(model, [trace_module]) as td:
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
        fc_vec = td[trace_module].output[0, last_pos, :].detach()
        fc_sum = fc_vec.to(torch.float64) if fc_sum is None else fc_sum + fc_vec
        n_examples += 1
        current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
        pbar.set_description(f'PPL: {current_ppl:.4f}')
    avg_fc = (fc_sum / n_examples).to(torch.float32)
    return avg_fc.squeeze()

def evaluate_llma_classification_batch_db14(model, eval_dataset, tokenizer, batch_size: int=32, device: str='cuda', max_new_tokens: int=5):
    label_mapping = {0: 'Company', 1: 'EducationalInstitution', 2: 'Artist', 3: 'Athlete', 4: 'OfficeHolder', 5: 'MeanOfTransportation', 6: 'Building', 7: 'NaturalPlace', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'WrittenWork'}
    model.to(device).eval()
    total, correct, total_conf = (0, 0, 0.0)
    fc_vals_correct = []
    loop = tqdm(range(0, len(eval_dataset), batch_size), desc='Batches')
    for start in loop:
        batch = eval_dataset[start:start + batch_size]
        texts = batch['text']
        labels = batch['label']
        prompts, gold_labels = ([], [])
        for text, lab_i in zip(texts, labels):
            gold = label_mapping[int(lab_i)]
            gold_labels.append(gold)
            prompts.append('Choose from one of these categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork. Be careful distinguishing between similar categories.\n\n{{ Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.:Company}}\n\n{{ Dubai Gem Private School (DGPS) is a British school located in the Oud Metha area of Dubai United Arab Emirates. Dubai Gem Nursery is located in Jumeirah. Together the institutions enroll almost 1500 students aged 3 to 18.:EducationalInstitution}}\n\n{{ Martin Marty McKinnon (born 5 July 1975 in Adelaide) is a former Australian rules footballer who played with Adelaide Geelong and the Brisbane Lions in the Australian Football League (AFL).McKinnon was recruited by Adelaide in the 1992 AFL Draft with their first ever national draft pick. He was the youngest player on Adelaide\'s list at the time and played for Central District in the SANFL when not appearing with Adelaide.:Athlete}}\n\n{{ The Wedell-Williams XP-34 was a fighter aircraft design submitted to the United States Army Air Corps (USAAC) before World War II by Marguerite Clark Williams widow of millionaire Harry P. Williams former owner and co-founder of the Wedell-Williams Air Service Corporation.:MeanOfTransportation}}\n\n{{"{}":'.format(text))
        enc = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
        orig_lens = enc['attention_mask'].sum(dim=1)
        gen, step0_conf, step0_fc = manual_generate_llma_batch_insert(model, enc['input_ids'], enc['attention_mask'], orig_lens, max_new_tokens=max_new_tokens)
        for i, gold in enumerate(gold_labels):
            gen_text = tokenizer.decode(gen[i, orig_lens[i]:], skip_special_tokens=True)
            pred = gen_text.split('}')[0].strip('"').strip()
            total += 1
            if pred == gold:
                correct += 1
                fc_vals_correct.append(step0_fc[i].squeeze())
            gold_tok = tokenizer.encode(gold, add_special_tokens=False)[0]
            total_conf += step0_conf[i, gold_tok].item()
        loop.set_description(f'Accuracy: {100 * correct / total:.1f}%')
    if total == 0:
        return (0.0, 0.0, torch.tensor([]))
    accuracy = round(correct / total, 4)
    mean_conf = round(total_conf / total, 4)
    fc_vals = torch.stack(fc_vals_correct) if fc_vals_correct else torch.tensor([])
    return (accuracy, mean_conf, fc_vals)

def evaluate_llma_mmlu(model, eval_dataset, tokenizer, max_samples=50):
    """
    Zero-shot evaluation on the MMLU multiple-choice benchmark.

    Args:
        model:      The (masked) LlamaForCausalLM to evaluate.
        eval_dataset: A HuggingFace Dataset with fields:
                      - 'question': str
                      - 'choices':  List[str]
                      - 'answer':   int  (index of the correct choice)
        tokenizer:  The corresponding tokenizer.
        max_samples:Max number of examples to score.

    Returns:
        accuracy:   float (correct / total)
    """
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    device = next(model.parameters()).device
    model.eval()
    total = 0
    correct = 0
    for ex in eval_dataset:
        prompt = f"Question: {ex['question']}\n"
        for idx, choice in enumerate(ex['choices']):
            prompt += f'{chr(65 + idx)}. {choice}\n'
        prompt += 'Answer:'
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=1, do_sample=False)
        gen = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
        if len(gen) and gen[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            pred_idx = ord(gen[0]) - 65
        else:
            pred_idx = -1
        if pred_idx == ex['answer']:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def evaluate_llma_classification_batch_emotions(model, eval_dataset, tokenizer, batch_size: int=64, device: str='cuda', max_new_tokens: int=5):
    label_mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    model.to(device).eval()
    total, correct, total_conf = (0, 0, 0.0)
    fc_vals_correct = []
    loop = tqdm(range(0, len(eval_dataset), batch_size), desc='Batches')
    for start in loop:
        batch = eval_dataset[start:start + batch_size]
        texts = batch['text']
        labels = batch['label']
        prompts, gold_labels = ([], [])
        for text, lab_i in zip(texts, labels):
            gold = label_mapping[int(lab_i)]
            gold_labels.append(gold)
            prompts.append('Choose from one of these: anger, fear, joy, love, sadness, surprise\n     {{"I can\'t believe how wonderful this day has been!":joy}}\n     {{"Missing you more with each passing day":sadness}}\n     {{"How dare they treat me like this!":anger}}\n     {{"I\'m getting butterflies just thinking about tomorrow":fear}}\n     {{"You mean everything to me":love}}\n     {{"I didn\'t expect this to happen at all":surprise}}\n     {{"{}":'.format(text))
        enc = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
        orig_lens = enc['attention_mask'].sum(dim=1)
        gen, step0_conf, step0_fc = manual_generate_llma_batch_insert(model, enc['input_ids'], enc['attention_mask'], orig_lens, max_new_tokens=max_new_tokens)
        for i, gold in enumerate(gold_labels):
            gen_text = tokenizer.decode(gen[i, orig_lens[i]:], skip_special_tokens=True)
            pred = gen_text.split('}')[0].strip('"').strip()
            total += 1
            if pred == gold:
                correct += 1
                fc_vals_correct.append(step0_fc[i].squeeze())
            gold_tok = tokenizer.encode(gold, add_special_tokens=False)[0]
            total_conf += step0_conf[i, gold_tok].item()
        loop.set_description(f'Accuracy: {100 * correct / total:.1f}%')
    accuracy = round(correct / total, 4)
    mean_conf = round(total_conf / total, 4)
    fc_vals = torch.stack(fc_vals_correct) if fc_vals_correct else torch.tensor([])
    return (accuracy, mean_conf, fc_vals)

def evaluate_llma_classification_ag_news(model, eval_dataset, tokenizer, batch_size: int=32, device: str='cuda', max_new_tokens: int=5):
    label_mapping = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    model.to(device).eval()
    total, correct, total_conf = (0, 0, 0.0)
    fc_vals_correct = []
    loop = tqdm(range(0, len(eval_dataset), batch_size), desc='Batches')
    for start in loop:
        batch = eval_dataset[start:start + batch_size]
        texts = batch['text']
        labels = batch['label']
        prompts, gold_labels = ([], [])
        for text, lab_i in zip(texts, labels):
            gold = label_mapping[int(lab_i)]
            gold_labels.append(gold)
            prompts.append('Choose from one of these categories: World, Sports, Business, Sci/Tech. Be careful distinguishing between similar topics.\n\n            {{ China unveils new space station module set for launch next year:World}}\n\n            {{ Lionel Messi nets two goals as PSG defeat Marseille 3-1:Sports}}\n\n            {{ Tesla shares jump 8% after delivery numbers beat expectations:Business}}\n\n            {{ Researchers develop a new algorithm to speed up quantum error correction:Sci/Tech}}\n\n            {{"{}":'.format(text))
        enc = tokenizer(prompts, padding=True, return_tensors='pt').to(device)
        orig_lens = enc['attention_mask'].sum(dim=1)
        gen, step0_conf, step0_fc = manual_generate_llma_batch_insert(model, enc['input_ids'], enc['attention_mask'], orig_lens, max_new_tokens=max_new_tokens)
        for i, gold in enumerate(gold_labels):
            gen_text = tokenizer.decode(gen[i, orig_lens[i]:], skip_special_tokens=True)
            pred = gen_text.split('}')[0].strip('"').strip()
            total += 1
            if pred == gold:
                correct += 1
                fc_vals_correct.append(step0_fc[i].squeeze())
            gold_tok = tokenizer.encode(gold, add_special_tokens=False)[0]
            total_conf += step0_conf[i, gold_tok].item()
        loop.set_description(f'Accuracy: {100 * correct / total:.1f}%')
    accuracy = round(correct / total, 4)
    mean_conf = round(total_conf / total, 4)
    fc_vals = torch.stack(fc_vals_correct) if fc_vals_correct else torch.tensor([])
    return (accuracy, mean_conf, fc_vals)
from rouge_score import rouge_scorer
from collections import Counter
import re
import torch

def evaluate_llma_language_modeling(model, eval_dataset, tokenizer, max_samples=5000, verbose=False, log_every=10):
    """
    Evaluate the perplexity of the model on a language-modeling task.

    Args:
        model:          The model to evaluate.
        eval_dataset:   The dataset to evaluate on.
        tokenizer:      The tokenizer to use.
        max_samples:    Maximum number of samples to evaluate.
        verbose:        If True, print running perplexity every `log_every` steps.
        log_every:      How often to print progress when verbose=True.

    Returns:
        float: The average perplexity across samples.
    """
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    model.eval()
    for i in range(len(eval_dataset)):
        example = eval_dataset[i]
        inputs = tokenizer(example['text'], return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            loss = model(**inputs, labels=inputs['input_ids']).loss
        seq_len = inputs['input_ids'].size(1)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
        if verbose and (i + 1) % log_every == 0:
            running_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
            print(f'{i + 1}/{len(eval_dataset)} — running perplexity: {running_ppl:.4f}', end='\r')
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print(f'\nFinal perplexity: {perplexity:.4f}')
    return perplexity

def reset_llma(model):
    model.model.mask_layer.lower_bound = torch.tensor(float('inf')).to(device)
    model.model.mask_layer.upper_bound = torch.tensor(float('-inf')).to(device)
    return model

def mask_range_llma(model, mask, fc_vals, replacement, tao):
    fc_vals = np.array(fc_vals)
    mean = torch.tensor(np.mean(fc_vals, axis=0))
    std = torch.tensor(np.std(fc_vals, axis=0))
    mask = mask.to(torch.bool)
    lower_bound = torch.full_like(mean, torch.inf)
    lower_bound[~mask] = mean[~mask] - tao * std[~mask]
    upper_bound = torch.full_like(mean, -torch.inf)
    upper_bound[~mask] = mean[~mask] + tao * std[~mask]
    model.model.mask_layer.lower_bound = lower_bound.to(device)
    model.model.mask_layer.upper_bound = upper_bound.to(device)
    return model

def manual_generate_llma_batch_insert(model: nn.Module, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, orig_lens: torch.LongTensor, max_new_tokens: int=5) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Same signature as before, but the final tensor keeps the shape
    (B, max_orig_len + max_new_tokens) and new tokens are *inserted*
    at orig_len + step rather than tacked to the end.
    """
    device = input_ids.device
    B = input_ids.size(0)
    pad_token_id = getattr(model.config, 'pad_token_id', 0)
    max_orig_len = input_ids.size(1)
    full_len = max_orig_len + max_new_tokens
    generated = torch.full((B, full_len), pad_token_id, dtype=input_ids.dtype, device=device)
    generated[:, :max_orig_len] = input_ids
    full_mask = torch.zeros_like(generated, dtype=attention_mask.dtype)
    full_mask[:, :max_orig_len] = attention_mask
    insert_pos = orig_lens.clone()
    with torch.no_grad(), nethook.TraceDict(model, ['model.mask_layer']) as ret:
        logits0 = model(input_ids=input_ids, attention_mask=attention_mask).logits
        next_logits = logits0[torch.arange(B), orig_lens - 1, :]
        step0_conf = torch.softmax(next_logits, dim=-1).cpu()
        next_tokens = torch.argmax(next_logits, dim=-1)
        per_layer = [layer_out[torch.arange(B, device=device), orig_lens - 1, :].detach().cpu() for layer_out in (ret[k].output for k in ret)]
        step0_fc_vals = torch.stack(per_layer, dim=1)
    generated[torch.arange(B), insert_pos] = next_tokens
    full_mask[torch.arange(B), insert_pos] = 1
    insert_pos += 1
    for _ in range(max_new_tokens - 1):
        active_len = insert_pos.max().item()
        with torch.no_grad():
            logits = model(input_ids=generated[:, :active_len], attention_mask=full_mask[:, :active_len]).logits
            last_pos = insert_pos - 1
            next_logits = logits[torch.arange(B, device=device), last_pos, :]
            next_tokens = torch.argmax(next_logits, dim=-1)
        generated[torch.arange(B), insert_pos] = next_tokens
        full_mask[torch.arange(B), insert_pos] = 1
        insert_pos += 1
    return (generated, step0_conf, step0_fc_vals)
