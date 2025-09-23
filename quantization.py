"""
QUANTIZATION
Conversion from higher memory format to a lower memory format.
Value of weights is stored as (for example) 32 bits (full precision/single precision), we can convert it to int8 (or FP16 - half precision, or any other) and then use the model
cost and resources go down
very useful for inference (faster and cheaper)
useful for less powerful devices like cell phone and other edge devices
we can fine tune quantized models, but there is loss of info and hence loss of accuracy
 

CALIBRATION - squeezing higher range to lower range
Symmetric Quantization 
e.g. batch norm - all weights are 0 centred
symmetric unsigned int8 quantization - we have floating point numbers [0.0, 1000.0] (for larger model, maybe 32 bits single precision floating point) - 1 sign, 7 exponent, 23 fraction (mantissa)
unsigned int8 would be [0, 255].
we can use min max scaler
scale_factor = x_max - x_min / q_max - q_min [1000 - 0 / 255 - 0]
x_new = round(x_old/scale_factor)

Asymmetric Quantization
symmetric uint8  [-20.0, 1000.0] -> values can be left or right skewed 
if we convert it to [0, 255]
scale facto = 4.0
x_min_new = -20/4 = -5 (it is signed) -> this number is called zero point
x_new = round(x_old/scale_factor) + zero_point

Post Training quantization (PTQ) - already have pretrained model -> apply calibration to weights and save it as quantized model
    - loss of info
Quantization aware training (QAT) - take the trained model, perform calibration, perform fine tuning on new training data.
    - trained on new precision values [done in finetune.py]
"""
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import torch
import gc

# QUANTIZATION - Post Training quantization
model_id = "meta-llama/Llama-3.1-8B"
# model_id = "openlm-research/open_llama_3b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=False)

# Load model in 16-bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",       
    dtype=torch.float16
)
print('Model loaded in 16-bit')
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"System RAM used (approx): {torch.cuda.memory_allocated()/1024**3 + torch.cuda.memory_reserved()/1024**3:.2f} GB")
# Test tokenization and generation
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print('16 bit result: ', tokenizer.decode(outputs[0], skip_special_tokens=True))
del model
torch.cuda.empty_cache() 
gc.collect()
torch.cuda.synchronize()

# Load model in 8-bit
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    # llm_int8_enable_fp32_cpu_offload=True
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config_8bit
)
print("Model loaded in 8-bit.")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"System RAM used (approx): {torch.cuda.memory_allocated()/1024**3 + torch.cuda.memory_reserved()/1024**3:.2f} GB")
outputs = model_8bit.generate(**inputs, max_new_tokens=50)
print('8-bit result: ', tokenizer.decode(outputs[0], skip_special_tokens=True))
del model_8bit
torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()

# Load model in 4-bit
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    # llm_int8_enable_fp32_cpu_offload=True
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config_4bit
)

print("Model loaded in 4-bit.")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"System RAM used (approx): {torch.cuda.memory_allocated()/1024**3 + torch.cuda.memory_reserved()/1024**3:.2f} GB")
outputs = model_4bit.generate(**inputs, max_new_tokens=50)
print('4-bit result: ', tokenizer.decode(outputs[0], skip_special_tokens=True))

# Integrate it with langchain

pipe = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=200
)
llm = HuggingFacePipeline(pipeline=pipe)
prompt = PromptTemplate.from_template("Answer the question: {query}")
chain = prompt | llm

query = "Explain the basics of Pokémon in simple terms."
print(chain.invoke(query))
del model_4bit