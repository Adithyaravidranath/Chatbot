# Qwen2.5 LoRA SFTTrainer Chatbot

This project demonstrates Supervised Fine-Tuning (SFT) of the Qwen2.5-0.5B-Instruct Large Language Model using the LoRA (Low-Rank Adaptation) technique. The chatbot is trained on a custom instruction dataset stored in JSONL format and optimized for efficient training with limited computational resources.

## Features

* Fine-tunes Qwen2.5-0.5B-Instruct using LoRA.
* Supports custom conversational datasets in JSONL format.
* Converts chat messages into User/Assistant training format.
* Uses TRL's SFTTrainer for supervised instruction tuning.
* Memory-efficient training with PEFT (Parameter-Efficient Fine-Tuning).
* Saves the fine-tuned model and tokenizer for chatbot inference.

## Training Pipeline

1. Load custom instruction dataset.
2. Format conversations into chatbot training text.
3. Load Qwen2.5 tokenizer and model.
4. Apply LoRA adapters to attention layers.
5. Fine-tune using SFTTrainer.
6. Save the trained model and tokenizer.

## Technologies Used

* Python
* Hugging Face Transformers
* TRL (Transformer Reinforcement Learning)
* PEFT (LoRA)
* PyTorch
* Datasets Library

## Model

**Base Model:** Qwen2.5-0.5B-Instruct

Output

The project produces a fine-tuned Qwen2.5 chatbot capable of answering user queries in a conversational format. After training, the model and tokenizer are saved in the chatbot_model directory and can be loaded for real-time chatbot interactions.

Chatbot Inference

The saved model can be loaded using the provided inference script to create an interactive command-line chatbot. Users can enter questions, and the chatbot generates responses based on the knowledge learned during fine-tuning.

Features of the Chatbot:
1.Interactive question-answering interface.
2.Generates responses using the fine-tuned Qwen2.5 model.
3.Supports continuous conversation until the user exits.
4.Uses temperature-based sampling for more natural responses.
5.Runs locally without requiring additional training.

### Example

```text
You: Hello
Bot: Hello! How can I assist you today?

You: What is machine learning?
Bot: Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.

You: exit
```

## Project Structure

```text
├── chatbot_instruction_dataset.jsonl
├── model trainer.py
├── chatbot.py
├── chatbot_model/
├── README.md
└── requirements.txt
```

## Future Improvements

* Train on larger conversational datasets.
* Add evaluation metrics for chatbot performance.
* Deploy as a web application.
* Experiment with larger Qwen models and advanced fine-tuning techniques.
