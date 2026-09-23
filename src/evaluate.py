from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "adapter" / "mistral-qlora-adapter"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    return model, tokenizer


def generate_answer(model, tokenizer, question):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def main():
    model, tokenizer = load_model()

    questions = [
        "What does stochastic mean?",
        "What is Python?",
        "What is machine learning?",
    ]

    for i, question in enumerate(questions, start=1):
        answer = generate_answer(
            model,
            tokenizer,
            question,
        )

        print(f"\n--- Example {i} ---")
        print("Question:", question)
        print("Answer:", answer)


if __name__ == "__main__":
    main()