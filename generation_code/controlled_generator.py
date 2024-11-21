from tqdm import tqdm
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore')

import os
import json
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import re
import textwrap

from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import get_random_prompt_xsum
from torch.utils.data import default_collate
from transformers import BatchEncoding

import torch

DEVICE = "cuda:2"


class ControlledModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.logits_mask = None
    
    def set_logits_mask(self, logits_mask):
        if len(logits_mask.shape) == 2:
            logits_mask = logits_mask.unsqueeze(1).to(self.device)
        self.logits_mask = logits_mask
        print("- successfully set logits' control mask!")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            num_logits_to_keep= 0,
            **loss_kwargs,
            ):

        base_model_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **loss_kwargs
        )

        controlled_logits = base_model_out.logits * self.logits_mask
        return CausalLMOutputWithPast(
            loss = base_model_out.loss,
            logits = controlled_logits,
            past_key_values=base_model_out.past_key_values,
            hidden_states=base_model_out.hidden_states,
            attentions=base_model_out.attentions
        )


class GenerationDataset(Dataset):
    def __init__(self, data, prompt_fn, tokenizer=None):
        self.data = data
        self.prompt_fn = prompt_fn
        self.tokenizer = tokenizer
        self.max_length = 512

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        _data = self.data.iloc[idx]
        title = _data.title
        doc_id = _data.id
        real_article = _data.real_article
        prompt = self.prompt_fn(title, informed=True)
        prompt_str = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        return {
            "model_inputs": self.tokenizer(prompt_str, max_length=self.max_length, padding="max_length", return_tensors="pt"),
            "real_article": real_article,
            "prompt_str": prompt_str,
            "title": title,
            "doc_id": doc_id
            }


def get_tokens_data(data: list, tokenizer):
    tok_ids = tokenizer(data).input_ids
    return tok_ids


def get_logits_mask(tokenizer, real_data, vocab_size, tokenizer_space_str=None):
    real_tokens = get_tokens_data(real_data, tokenizer)
    real_tokens = sum(real_tokens, [])
    real_tokens = list(set(real_tokens))
    num_real_tokens = len(real_tokens)
    real_tokens += list(tokenizer.get_added_vocab().values())   # re-add special tokens (i.e., added vocab)

    if tokenizer_space_str is not None:
        print(f"!! tokenizer space string set to: {tokenizer_space_str} !!")
        space_tok_id = tokenizer.vocab[tokenizer_space_str]
        real_tokens += [space_tok_id]
    real_tokens = sorted(real_tokens)
    
    print(f"- real tokens selected: {num_real_tokens} (+ {len(real_tokens) - num_real_tokens} model reserved toks)")


    logits_mask = torch.zeros(size=(1, vocab_size))
    logits_mask[:, [real_tokens]] = True

    return logits_mask, real_tokens


def postprocess_text(text):
    try:
        text = re.sub(r"\s+", " ", text)
    except:
        pass
    
    return text


class CustomCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, data):
        # batch = [elem["model_inputs"] for elem in data]
        # prompts = [elem["prompt_str"] for elem in data]
        # titles = [elem["titles"] for elem in data]
        # doc_ids = [elem["doc_id"] for elem in data]

        # input_ids = torch.vstack([b["input_ids"] for b in batch])
        # attention_masks = torch.vstack([b["attention_mask"] for b in batch])
        # return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_masks}), prompts, titles, doc_ids
        
        batch, real_article, prompts, titles, doc_ids = zip(
        *[(elem["model_inputs"], elem["real_article"], elem["prompt_str"], elem["title"], elem["doc_id"]) for elem in data]
        )
    
        # Stack tensors for input_ids and attention_mask
        inputs = {key: torch.vstack([b[key] for b in batch]) for key in ["input_ids", "attention_mask"]}
        return BatchEncoding(inputs), list(real_article), list(prompts), list(titles), list(doc_ids)
    


def main(args):
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    data_path = "data/data_2024_11_08/generation_output_llama-3.1-70b-instruct-hf_xsum_temp0.8_informed.jsonl"
    split_path = os.path.join("data/splits", f"llama-3.1-8b-instruct-hf_xsum_informed.split.{str(args.num_samples)}.json" if "llama-3.1-8b" in model_name.lower() else f"llama-3.1-70b-instruct-hf_xsum_informed.split.{str(args.num_samples)}.json")
    df = pd.read_json(data_path, lines=True)
    selected = json.load(open(split_path))

    outfile = f"generation_output_{model_name.split('/')[-1].lower()}_xsum_temp{args.temperature}.controlled{str(args.controlling_docs)}.testonly.jsonl"

    # df_te = df[df["id"].isin(selected["te"])]
    df_tr = df[df["id"].isin(selected["tr"])]

    # print(f"-  number of documents in the test set split {args.num_samples}: {len(df_te)}")

    # we sample the controlling documents from the training split
    real_articles = df_tr.real_article.sample(n=args.controlling_docs, random_state=42).to_list()

    model = ControlledModel.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # dataset = GenerationDataset(df_te.title.to_list(), get_random_prompt_xsum, tokenizer)
    dataset = GenerationDataset(df, get_random_prompt_xsum, tokenizer)

    logits_mask, selected_newhead_toks = get_logits_mask(tokenizer, real_articles, vocab_size=model.vocab_size)
    model.set_logits_mask(logits_mask)

    dataloader = DataLoader(dataset=dataset, batch_size=args.batch, collate_fn=CustomCollate(tokenizer))

    print(model.generation_config)

    with torch.no_grad():
        for batch, real_articles, prompts, titles, doc_ids in tqdm(dataloader):
            generated_ids = model.generate(**batch.to(DEVICE), max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, temperature=args.temperature)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            generated_text = [postprocess_text(t.split("assistant\n\n")[-1]) for t in generated_text]

            # print(f"\n({i}) Title: {df_te.iloc[i].title}")
            # print(textwrap.fill(generated_text, width=75))

            for t, real_article, prompt, title, doc_id in zip(generated_text, real_articles, prompts, titles, doc_ids):
                with open(outfile, "a") as jf:
                    to_dump = {
                        "prompt": prompt,
                        "generated_text": t,
                        "real_article": real_article,
                        "id": str(doc_id),
                        "title": title,
                        "source": "xsum" 
                    }

                    jf.write(json.dumps(to_dump) + "\n")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--controlling_docs", type=int, default=250)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--batch", default=4, type=int)
    args = parser.parse_args()
    main(args)


