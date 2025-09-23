"""
We get weights of a LLM trained on huge data. fine tune on new data on all params. major challenge -> expensive
then we can perform domain specific fine tuning (for medical data etc) 
or we can perform specific task fine tuning (for Q&A chatbot, document retrieval etc)

LoRA (Low Rank Adaptation of LLM) - instead of updating all weight, it will track the changes in weights based on fine tuning (same size matrix) and based on that it will update weights
the tracked weights are stored as 2 vectors which is created with matrix decomp (the vectors can be cross product-ed to reform the matrix)
there is loss in precision but the resources needed are much lower.
W0 + W_delta = W0 + BxA
the parameters increase linearly with the rank increase (instead of polynomial)
high ranks are used - if we want more complex model. if we need a simple LLM, we can use smaller ranks

QLora (Quantized LoRA) - All params stored in W_delta, we store it in lower precision (saves even more resources)

PEFT - parameter efficient fine tuning
TRL - Transformers Reinforcement Learning
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer

import torch


def preprocess(example):
    # Get text and label
    text = example.get("text") or example.get("prompt", "")
    label = "positive" if example.get("label", 1) == 1 else "negative"

    # Build prompt and completion
    prompt = f"Review: {text.strip()}\n"
    completion = f" {label.strip()}"

    return {"prompt": prompt, "completion": completion}

model_id = "meta-llama/Llama-3.1-8B"
# model_id = "openlm-research/open_llama_3b" 

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=False)
tokenizer.pad_token = tokenizer.eos_token # need to add this because it doesnt have a pad token by default

# Load model in 4-bit
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config_4bit
)

model = prepare_model_for_kbit_training(model,
    use_gradient_checkpointing=True,  
) # THIS IS SUPER IMPORTANT FOR QLoRA, it freezes things that are not required to be updated

# LoRA config
lora_config = LoraConfig(
    r=16,                       # Rank for matrix decomposition
    lora_alpha=32,              # alpha is the scaling factor using in it
    target_modules=["q_proj", "v_proj"],  # Often, LoRA is applied to Q and V because they carry most of the task-specific variation
    lora_dropout=0.05,          # Dropout applied only on the LoRA branch, not the frozen model
    task_type="CAUSAL_LM"       # Specifies what task the model is being fine-tuned for
)

model_peft = get_peft_model(model, lora_config)

# Dataset (existing dataset)
dataset = load_dataset("imdb", split="train[:200]")
dataset = dataset.map(preprocess)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Supervised Fine-Tuning
training_args = TrainingArguments(
    output_dir="./tmp_unused",  
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no",
    fp16=False
)
trainer = SFTTrainer(
    model=model_peft,
    train_dataset=train_dataset,
    args=training_args,
)
trainer.train()

# generating using finetuned model
prompt = "I liked the movie and the acting was great"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model_peft.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated:", generated_text)

print()
# Dataset (custom dataset)
custom_data = [
    {"prompt": "Who is Pikachu?", "completion": "Pikachu is an Electric-type Pokémon."},
    {"prompt": "What is Bulbasaur?", "completion": "Bulbasaur is a Grass/Poison-type Pokémon."}
]

custom_dataset = Dataset.from_list(custom_data)
train_dataset = custom_dataset
eval_dataset = custom_dataset
print('Custom Dataset loaded\n')

# Same steps
model_peft = get_peft_model(model, lora_config)
training_args = TrainingArguments(
    output_dir="./tmp_unused", 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no",
    fp16=False
)
trainer = SFTTrainer(
    model=model_peft,
    train_dataset=train_dataset,
    args=training_args,
)
trainer.train()
prompt = "Tell me about pikachu"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model_peft.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    use_cache=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated:", generated_text)