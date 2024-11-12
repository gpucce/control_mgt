import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.integrations import WandbCallback 
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import PeftModel
from llm2vec import LLM2Vec


def main():
    device = "cuda"

    base_model_id = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
    model_id = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # Loading base model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
    config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True, config=config, torch_dtype=torch.bfloat16, device_map="cuda" if device == "cuda" else "cpu")
    base_model = PeftModel.from_pretrained(base_model, base_model_id)
    base_model = base_model.merge_and_unload()
    # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
    base_model = PeftModel.from_pretrained(base_model, model_id)
    
    documents = ["this is a document"]
    labels = torch.tensor(1)
    llm2vec_model_inputs = llm2vec_tokenize(tokenizer, documents)

    model = LLM2VecForSequenceClassification(base_model=base_model, num_labels=2, tokenizer=tokenizer).to(device)
    my = model(**llm2vec_model_inputs.to(device), labels=labels.to(device))

    # llm2vec_model = LLM2Vec(base_model, tokenizer, pooling_mode="mean", max_length=512)
    # ref = llm2vec_model.encode(documents)


def _convert_to_str(tokenizer, instruction, text, max_length=512, doc_max_length=512):
    tokenized_q = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    tokenized_q_length = len(tokenized_q["input_ids"][0])

    while tokenized_q_length > doc_max_length:
        reduction_ratio = doc_max_length / tokenized_q_length
        reduced_length = int(len(text.split()) * reduction_ratio)
        text = " ".join(text.split()[:reduced_length])
        tokenized_q = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        tokenized_q_length = len(tokenized_q["input_ids"][0])

    return (
        f"{instruction.strip()} !@#$%^&*(){text}"
        if instruction
        else f"!@#$%^&*(){text}"
    )


def _tokenize(tokenizer, texts, max_length=512):
    texts_2 = []
    original_texts = []
    for text in texts:
        t = text.split("!@#$%^&*()")
        texts_2.append(t[1] if len(t) > 1 else "")
        original_texts.append("".join(t))

    original = tokenizer(
        original_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length= max_length,
    )
    embed_mask = None
    for t_i, t in enumerate(texts_2):
        ids = tokenizer(
            [t],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        if embed_mask is None:
            e_m = torch.zeros_like(original["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                    len(ids["input_ids"][0])
                )
            embed_mask = e_m.unsqueeze(0)
        else:
            e_m = torch.zeros_like(original["attention_mask"][t_i])
            if len(ids["input_ids"][0]) > 0:
                e_m[-len(ids["input_ids"][0]) :] = torch.ones(
                    len(ids["input_ids"][0])
                )
            embed_mask = torch.cat((embed_mask, e_m.unsqueeze(0)), dim=0)

    original["embed_mask"] = embed_mask
    return original


def llm2vec_tokenize(tokenizer, texts, max_length=512, doc_max_length=400):
    if isinstance(texts[0], str) and isinstance(texts[-1], int):
        texts = [texts]
    if isinstance(texts[0], str):
        texts = [[""] + [text] for text in texts]
    
    concat_input_texts = []
    for text in texts:
        assert (isinstance(text[0], str) and isinstance(text[1], str))
        concat_input_texts.append(
            _convert_to_str(tokenizer=tokenizer, instruction=text[0], text=text[1], max_length=max_length, doc_max_length=doc_max_length)
        )
    texts = concat_input_texts

    model_inputs = _tokenize(tokenizer, texts, max_length=max_length)
    
    return model_inputs


class LLM2VecForSequenceClassification(LLM2Vec):
    def __init__(
        self,
        base_model,
        num_labels,
        tokenizer,
        pooling_mode="mean",
        max_length=512,
        doc_max_length=400,
        skip_instruction=True
    ):
        super().__init__(
            model=base_model,
            tokenizer=tokenizer,
            pooling_mode=pooling_mode,
            skip_instruction=skip_instruction,
            max_length=max_length,
            doc_max_length=doc_max_length
            )
        self.model = base_model
        self.config = self.model.config
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_labels, dtype=self.model.dtype)
    
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            embed_mask=None,
            labels=None,
            output_hidden_states=None,
            output_attentions=None,
            **kwargs
    ):
        reps = self.model(input_ids, attention_mask)
        pooled = self.get_pooling(
            {"input_ids": input_ids, "attention_mask": attention_mask, "embed_mask": embed_mask},
              reps.last_hidden_state)
        
        logits = self.classifier(pooled)
        loss = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pooled if output_hidden_states else None,
            attentions=None
        )
        

if __name__ == "__main__":
    main()
