import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from google.colab import drive

# 1. Mount your Google Drive
drive.mount('/content/drive')

# 2. Define the save directory and model name
save_directory = "/content/drive/MyDrive/my_models/phi-3-mini"
model_name = "microsoft/Phi-3-mini-4k-instruct"

# 3. Configure Quantization to save memory during the loading process
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 4. Download and load the model and tokenizer using the memory-saving config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. Save the full model to your Drive
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 6. Configure Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 7. Load Base Model & Tokenizer from Your Google Drive
model_path = "/content/drive/MyDrive/my_models/phi-3-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# 8. Load your dataset
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# 9. Format dataset into a single "text" field
def format_example(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.map(format_example)

# 10. Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 11. Set up Training Arguments (SFTConfig = TrainingArguments + extras)
training_args = SFTConfig(
    output_dir="./email-finetune-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    fp16=True,
)

# 12. Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 13. Create Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    data_collator=data_collator
)

# 14. Start Training
trainer.train()

# 15. Save your fine-tuned adapter
output_dir = "./my-email-lora-adapter"
trainer.save_model(output_dir)
print(f"Model adapter saved to {output_dir}.")