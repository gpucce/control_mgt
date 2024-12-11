from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from trl import (
    DPOConfig,
    DPOTrainer,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datetime import datetime


def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

def prepare_model_for_LoRA_training(model, accelerator):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config, adapter_name="training_adapter")
    print_trainable_parameters(model)

    # Apply the accelerator. You can comment this out to remove the accelerator.
    model = accelerator.prepare_model(model)
    return model

def get_quant_model(model_name):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    return model

def get_accelerator():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    return accelerator

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    return tokenizer

if __name__ == "__main__":
    accelerator = get_accelerator()
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model = get_quant_model(model_name)

    # model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    dataset = load_dataset(
        "json",
        data_files="dpo_dataset/data/dataset-max-feature-difference-top-10_iter_1.zip",
        split="train")

    model = prepare_model_for_LoRA_training(model, accelerator)
    # Define the DPOConfig with your training parameters
    training_config = DPOConfig(
        output_dir="model",
        warmup_steps=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5.0e-6,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="model/logs",
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        do_eval=True,
        num_train_epochs=1,
        run_name=f"test-llama{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    # Initialize the DPOTrainer with the DPOConfig
    trainer = DPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
    trainer.save_state("dpo_dataset/model")