from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "/content/chatbot_instruction_dataset.jsonl"

# Load dataset
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)["train"]

# Use only first 1000 samples
dataset = dataset.select(range(min(1000, len(dataset))))

# Format dataset
def format_chat(example):
    text = ""

    for msg in example["messages"]:
        if msg["role"] == "user":
            text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            text += f"Assistant: {msg['content']}\n"

    return {"text": text}

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

# LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training args
training_args = TrainingArguments(
    output_dir="./chatbot_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=2
)

# New SFTTrainer API
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# Save
model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")

print("Training Complete")
