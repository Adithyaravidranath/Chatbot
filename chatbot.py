from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./chatbot_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def chat(question):
    prompt = f"User: {question}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response

while True:
    q = input("You: ")

    if q.lower() == "exit":
        break

    print("\nBot:", chat(q))
    print()
