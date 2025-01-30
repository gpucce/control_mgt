from collections import Counter
from nltk.util import ngrams
import torch

from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class ControlledModel(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.logits_mask = None
        self.control_fn = None
        self.control_fn = self.set_gen_mode()
        self.prev_gen = None
    
    def set_logits_mask(self, logits_mask):
        if len(logits_mask.shape) != 2:
            logits_mask = logits_mask.unsqueeze(1)
        self.logits_mask = logits_mask.to(self.device)
    
    def set_gen_mode(self, mode="vanilla", smooth=0.0, alpha_allowed=0.0):
        logits_mask = self.logits_mask
        self.smooth = smooth
        self.alpha_allowed = alpha_allowed

        if mode != "vanilla":
            logits_mask = logits_mask * alpha_allowed
            if "bigram" in mode:
                pass
            else:
                logits_mask = logits_mask + smooth

        if mode == "logits_dot_mask":
            control_fn = lambda x: x * logits_mask
        elif mode == "softmax_dot_mask":
            control_fn = lambda x: torch.nn.functional.softmax(x, dim=-1) * logits_mask
        elif mode == "softmax_dot_softmax":
            control_fn = lambda x: torch.nn.functional.softmax(x, dim=-1) * torch.nn.functional.softmax(logits_mask, dim=-1)
        elif mode == "logits_plus_mask":
            control_fn = lambda x: x + logits_mask
        elif mode == "softmax_plus_softmax":
            control_fn = lambda x: torch.nn.functional.softmax(x, dim=-1) + torch.nn.functional.softmax(logits_mask, dim=-1)
        elif mode == "softmax_plus_mask":
            control_fn = lambda x: torch.nn.functional.softmax(x, dim=-1) + logits_mask
        elif mode == "softmax_mean_mask":
            control_fn = lambda x: (torch.nn.functional.softmax(x, dim=-1) + logits_mask) / 2
        elif mode == "softmax_mean_softmax":
            control_fn = lambda x: (torch.nn.functional.softmax(x, dim=-1) + torch.nn.functional.softmax(logits_mask, dim=-1)) / 2
        elif mode == "bigram_logits_dot_mask":
            # TODO check this out
            # control_fn = lambda x, y: (x * self.logits_mask[y]).to_dense()
            control_fn = lambda x, y: x * (self.logits_mask[y].to_dense() + self.smooth)
        elif mode == "bigram_softmax_dot_mask":
            control_fn = lambda x, y: torch.nn.functional.softmax(x, dim=-1) * (self.logits_mask[y].to_dense() + self.smooth)
        elif mode == "bigram_softmax_dot_softmax":
            control_fn = lambda x, y: torch.nn.functional.softmax(x, dim=-1) * torch.nn.functional.softmax((self.logits_mask[y].to_dense() + self.smooth), dim=-1)
        elif mode == "vanilla":
            control_fn = lambda x: x
        
        self.gen_mode = mode
        self.control_fn = control_fn
    
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

        if "bigram" not in self.gen_mode:
            controlled_logits = self.control_fn(base_model_out.logits)
        else:
            if self.prev_gen is None:
                # if it's first gen, we skip the conditioning part and store the softmax argmax as self.prev_gen
                # print("[hit first gen]")
                controlled_logits = base_model_out.logits
            else:
                # print(f"[hit cont gen {self.prev_gen}]")
                controlled_logits = self.control_fn(base_model_out.logits, y=self.prev_gen)
            
            self.prev_gen = torch.nn.functional.softmax(controlled_logits, dim=-1).argmax()
            
        return CausalLMOutputWithPast(
            loss = base_model_out.loss,
            logits = controlled_logits,
            past_key_values=base_model_out.past_key_values,
            # hidden_states=base_model_out.hidden_states,
            hidden_states=base_model_out.logits,    # TODO assigning base mode logits here just for debugging
            attentions=base_model_out.attentions
        )


def create_ngram_logits_mask(ngrams_count, n, vocab_size):
    # Step 1: Extract indices and values
    indices = torch.tensor(list(ngrams_count.keys()), dtype=torch.long).T  # Transpose to match sparse format
    values = torch.tensor(list(ngrams_count.values()), dtype=torch.float)

    # Step 2: Define the shape of the sparse tensor
    shape = tuple(vocab_size for i in range(n))

    # Step 3: Create the sparse tensor
    logits_mask = torch.sparse_coo_tensor(indices, values, size=shape)
    return logits_mask


def get_ngram_mask(real_data, tokenizer, vocab_size, n=2, verbose=False):
    bigram_counts = Counter()

    if verbose: print("- tokenizing data")
    docs = tokenizer(real_data).input_ids

    if verbose: print("- computing bigrams")
    for sent in docs:
        bigram_counts.update(ngrams(sent, n))
    
    if verbose: print("- creating logit mask")
    logits_mask = create_ngram_logits_mask(bigram_counts, n, vocab_size)
    
    return logits_mask, bigram_counts
