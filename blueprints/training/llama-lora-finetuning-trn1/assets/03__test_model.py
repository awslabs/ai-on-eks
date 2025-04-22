from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import argparse

# Set these as constants
TOKENIZER_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
SAMPLE_INDICES = [94, 99, 123]

def main():
    parser = argparse.ArgumentParser(description='Evaluate SQL translation models')
    parser.add_argument('--tuned-model', required=True, help='Path to tuned model')
    args = parser.parse_args()

    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    tuned_model = AutoModelForCausalLM.from_pretrained(args.tuned_model)

    # Prepare dataset
    dataset = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=23)
    eval_dataset = dataset.select(range(50000, 50500)).map(
        create_conversation,
        remove_columns=dataset.features,
        batched=False
    )

    # Run evaluation
    evaluate_models(eval_dataset, tokenizer, base_model, tuned_model, SAMPLE_INDICES)

def create_conversation(sample):
    system_msg = "You are a text-to-SQL translator. Generate SQL queries based on:\nSCHEMA:\n{schema}"
    return {
        "messages": [
            {"role": "system", "content": system_msg.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }

def evaluate_models(dataset, tokenizer, base_model, tuned_model, indices):
    for idx in indices:
        messages = dataset[idx]['messages'][:-1]

        # Tokenize input
        example = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        prompt_len = tokenized["input_ids"].size(1)

        # Clear screen and display
        subprocess.run("clear", shell=True)
        print(f"Evaluating sample {idx}")
        print(f"PROMPT:\n{example}\n")
        print("Generating output for the prompt using the base model and the new fine-tuned model. Please wait...\n\n")

        # Generate outputs
        base_output = base_model.generate(**tokenized, max_new_tokens=50, pad_token_id=128001)
        tuned_output = tuned_model.generate(**tokenized, max_new_tokens=50, pad_token_id=128001)

        # Display results
        print("BASE MODEL:\n", tokenizer.decode(base_output[0][prompt_len:]), "\n")
        print("FINE-TUNED MODEL:\n", tokenizer.decode(tuned_output[0][prompt_len:]), "\n")

        _ = input("Press <ENTER> to continue")

if __name__ == "__main__":
    main()
