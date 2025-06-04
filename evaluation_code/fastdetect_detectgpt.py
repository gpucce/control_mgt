# https://github.com/eric-mitchell/detect-gpt
import os
import json 
import transformers
import torch
import random
import datasets
import numpy as np
import re
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report
import tqdm 
from utils import *
import functools

DEVICE = "cuda"
pattern = re.compile(r"<extra_id_\d+>")

# Get the log likelihood of each text under the base_model
def get_ll(text):
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    return [get_ll(text) for text in texts]

def __threshold(num):
    return 0 if num < 0.5 else 1

def classification_report_detectgpt(real_preds, sample_preds):
    y_pred = list(map(__threshold,real_preds)) + list(map(__threshold,sample_preds))
    return classification_report(y_true=[0]*len(real_preds) + [1]*len(sample_preds), 
                                y_pred=y_pred, output_dict=True)

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def __load_base_model_and_tokenizer(name):
    print(f'Loading BASE model {args.base_model_name}...')
    base_model_kwargs = {}
    if 'gpt-j' in name or 'neox' in name:
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
    if 'gpt-j' in name:
        base_model_kwargs.update(dict(revision='float16'))
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, token=os.getenv("MY_HF_TOKEN"), torch_dtype=torch.bfloat16)
    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, token=os.getenv("MY_HF_TOKEN"))
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer

def __load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    try:
        mask_model.cpu()
    except NameError:
        pass
    base_model.to(DEVICE)
    
def __load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)

    base_model.cpu()
    mask_model.to(DEVICE)
        
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    
    return perturbed_texts
        
def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs

# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=30, prompt_tokens=30):
    # encode each text as a list of token ids
    all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    
    decoded = ['' for _ in range(len(texts))]

    # sample from the model until we get a sample with at least min_words words for each example
    # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = args.top_p
        elif args.do_top_k:
            sampling_kwargs['top_k'] = args.top_k
        min_length = 50
        outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
        decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1

    return decoded

def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=30)

        for o, s in zip(original_text, sampled_text):

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        __load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        __load_base_model()

    return data

def generate_data(dataset, key = "real"):

    data = dataset[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    # data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    # random.seed(0)
    # random.shuffle(data)

    #data = data[:5_000]

    tokenized_data = base_tokenizer(data, truncation=True, max_length=max_length)
    data = base_tokenizer.batch_decode(tokenized_data['input_ids'], skip_special_tokens=True)

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    # return generate_samples(data[:n_samples], batch_size=batch_size)
    return generate_samples(data, batch_size=batch_size)

def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    __load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    __load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'originals': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['originals'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['originals'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    fpr, tpr, roc_auc = get_roc_metrics(predictions['originals'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['originals'], predictions['samples'])
    print(classification_report_detectgpt(predictions['originals'], predictions['samples']))
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def get_data(datapath):
    if datapath.endswith(".zip"):
        data = pd.read_json(datapath, lines=True).to_dict(orient="records")
    else:
        data = json.load(open(datapath))
    return data


def main(args):
    global  base_model, mask_model, base_tokenizer, preproc_tokenizer, mask_tokenizer, FILL_DICTIONARY, mask_filling_model_name, n_samples, batch_size, n_perturbation_rounds, n_similarity_samples, data, max_length
    
    # with open(args.datapath, "r") as input_file:
    #     temp = json.loads(input_file.read())
    temp = get_data(args.datapath)

    if args.split_path is not None:
        test_split = json.load(open(args.split_path))["te"]
        data = pd.DataFrame(temp)
        data["doc-id"] = data["doc-id"].astype(int)
        data = data[data["doc-id"].isin(test_split)]
        temp = data.to_dict(orient="records")
        output_dir = os.path.join("evaluation_code", "evaluations", args.datapath.split("/")[-1].replace(".zip", ""), "detect-gpt_detector", args.target)
    else:
        output_dir = os.path.join("evaluation_code", "evaluations", *args.datapath.split("/")[2:-1], "detect-gpt_detector", args.target)
    
    dataset = datasets.Dataset.from_dict({
        'real': [el["human"] for el in temp],
        'sample': [el[args.target] for el in temp],
        'doc_id': [elem["doc-id"] for elem in temp],
    })
    
    del temp
    
    print(f"- Evaluation detect-gpt INFO" + "-" * 25)
    print(f"- storing results in: {output_dir}")
    print(f"- target: {args.target}")
    print(f"- num pairs: {dataset.num_rows}")
    print("-" * 45)

    ### detecion code starts
    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbations = int(args.n_perturbations)
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples
    max_length = args.max_length

    # generic generative model
    base_model, base_tokenizer = __load_base_model_and_tokenizer(args.base_model_name)

    # mask filling t5 model
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    print(f'Loading mask filling model {mask_filling_model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, torch_dtype=torch.bfloat16)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)

    __load_base_model()
    
    outputs = []
    
    for key in ['real', 'sample']: 
        data = generate_data(dataset, key)

        perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
        perturbation_mode = 'd' if args.no_normalization else 'z'
        output = run_perturbation_experiment(
            perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
        outputs.append(output)

    
    ### detection code ends
    metrics = classification_report_detectgpt(outputs[0]['predictions']['originals'], outputs[1]['predictions']['originals'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "clf_metrics.json"), "w") as jf:
        json.dump(metrics, jf)
    
    all_ids = dataset['doc_id'][:n_samples] + dataset['doc_id'][:n_samples]

    # store models output (scores/logits/probs) too! (e.g., {"p_0": soft_0, "p_1": soft_1})
    y_pred = list(map(__threshold, outputs[0]['predictions']['originals'])) + list(map(__threshold, outputs[1]['predictions']['originals']))
    y_true = [0]* len(outputs[0]['predictions']['originals']) + [1] * len(outputs[1]['predictions']['originals'])
    p_0 = outputs[0]['predictions']['originals'] + outputs[1]['predictions']['originals']
    print(len(y_pred), len(y_true), len(p_0), len(all_ids))

    df_preds = pd.DataFrame(data={"doc-id": all_ids, 
                             "y_pred": y_pred,
                             "y_true": y_true, 
                             "p_0": p_0,
                             "p_1": p_0
                                 })
    df_preds.to_csv(os.path.join(output_dir, "clf_preds.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/xsum-iter-1/llama-dpo-iter1/0130-2348/generations-0203_1752.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target", type=str, default="llama-dpo-iter1")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--split_path", type=str, default="data/xsum/splits/split.100000.json")
    
    parser.add_argument("--no_normalization", action='store_true')
    parser.add_argument('--dataset_key', type=str, default="real")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=22500)
    parser.add_argument('--n_perturbations', type=int, default="1")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="openai-community/gpt2")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    args = parser.parse_args()
    main(args)