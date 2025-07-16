#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # CHANGE THIS BASED ON YOUR AVAILABLE GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Gemma3ForConditionalGeneration
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import login
login(token = "") # INSERT HUGGINGFACE ACCESS TOKENS

torch.cuda.empty_cache()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CEFR classification with LLMs.")
    parser.add_argument("--prompt_template", type=str, required=True,
                        help="Path to the prompt template .txt file.")
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Hugging Face dataset ID or local path.")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face model ID.")
    parser.add_argument("--lang_prompts", action="store_true", help="Use language specific prompts.")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_id, split="train")

    # Load model and tokenizer/processor
    device = torch.device("cuda")
    print("CUDA device count:", torch.cuda.device_count())
    print("Model:", args.model_id)

    model_type = None
    tokenizer = None
    processor = None

    if "gemma-3-12b-it" in args.model_id:
        config = AutoConfig.from_pretrained(args.model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_id, config=config, torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        model_type = "gemma3"

    elif "EuroLLM-9B-Instruct" in args.model_id or "gemma-7b-it" in args.model_id:
        config = AutoConfig.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model_type = "llama"

    elif "aya-101" in args.model_id:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, trust_remote_code=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
        model_type = "aya-101"

    elif "Llama-3.1-8B" in args.model_id or "bloomz" in args.model_id:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model_type = "llama3"

    else:
        raise ValueError(f"Model {args.model_id} is not supported in this script.")

    recognized_labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    predictions = []
    gold_labels = []

    # Generate predictions
    counter = 0
    print("Model Type:", model_type)
    for sample in dataset:
        if ("+" in sample["cefr_level"]) and ("1" not in sample["cefr_level"] or "2" not in sample["cefr_level"]):
            print(f"Skipping entry {counter} due to '+' in cefr_level")
            continue

        text = sample["text"]
        gold_label = sample["cefr_level"]
        lang = sample["lang"]

        # Use language-specific CEFR descriptors if True, else use English
        if args.lang_prompts:
            prompt_template_file = args.prompt_template
            lang_prompt_file = prompt_template_file.replace('.txt','')
            lang_prompt_file = lang_prompt_file+'_'+lang+'.txt'
            print("Using language-specific CEFR descriptors: ", lang)
            with open(lang_prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        else:
            print("Using default English CEFR descriptors")
            with open(args.prompt_template, "r", encoding="utf-8") as f:
                prompt_template = f.read()

        user_prompt = prompt_template.replace("<<TEXT>>", text)

        if model_type == "gemma3":
            messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant for CEFR classification."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }]
            inputs = processor.apply_chat_template(messages, padding="longest", pad_to_multiple_of=8, add_generation_prompt=True, tokenize=True,return_dict=True, return_tensors="pt").to(device)
            inputs = {k: v.contiguous().to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                generation = generation[0][input_len:]
                decoded_output = processor.decode(generation, skip_special_tokens=True)


        elif model_type == "llama3" and "Llama" in args.model_id:
            messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant for CEFR classification."
            },

            {
                "role": "user",
                "content": user_prompt
            }]
            input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(device)
            terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            with torch.inference_mode():
                generation = model.generate(input_ids, eos_token_id=terminators, max_new_tokens=10, do_sample=False)
                response = generation[0][input_ids.shape[-1]:]
                decoded_output = tokenizer.decode(response, skip_special_tokens=True)

        elif model_type == "llama" and "EuroLLM-9B-Instruct" in args.model_id:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for CEFR classification.",
                },
                {
                    "role": "user", "content": user_prompt
                }]

            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_length = inputs.shape[-1]
            outputs = model.generate(inputs, max_new_tokens=10)
            generated_tokens = outputs[0][input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            decoded_output = decoded_output.strip()

        elif "aya-101" in args.model_id:
            print("Here")
            inputs = tokenizer(user_prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
            input_size = inputs["input_ids"].shape[1]
            outputs = model.generate(**inputs, do_sample=True, min_length=1, max_length=100)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        elif "bloomz" in args.model_id:
            inputs = tokenizer(user_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
            generated_tokens = outputs[0][input_ids.shape[-1]:]  # slice off the prompt
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        else:
            input_ids = tokenizer(user_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids, max_new_tokens=10)
            generated_tokens = outputs[0][input_ids['input_ids'].shape[-1]:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            decoded_output = decoded_output.strip()

        print(decoded_output)

        predicted_label = "UNKNOWN"
        for label in recognized_labels:
            if label in decoded_output:
                predicted_label = label
                break

        predictions.append(predicted_label)
        gold_labels.append(gold_label)
        counter += 1
        print(counter)

        torch.cuda.empty_cache()

    # Save predictions
    with open("predictions.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "lang", "format", "gold_label", "prediction"])
        for sample, pred in zip(dataset, predictions):
            writer.writerow([sample["text"], sample["lang"], sample["format"], sample["cefr_level"], pred])

    # Compute metrics
    valid_indices = [i for i, p in enumerate(predictions) if p != "UNKNOWN"]
    filtered_preds = [predictions[i] for i in valid_indices]
    filtered_gold = [gold_labels[i] for i in valid_indices]

    accuracy = accuracy_score(filtered_gold, filtered_preds)
    precision = precision_score(filtered_gold, filtered_preds, average="macro", labels=recognized_labels)
    recall = recall_score(filtered_gold, filtered_preds, average="macro", labels=recognized_labels)
    f1 = f1_score(filtered_gold, filtered_preds, average="macro", labels=recognized_labels)

    print("Accuracy:", accuracy)
    print("Precision (macro):", precision)
    print("Recall (macro):", recall)
    print("F1 (macro):", f1)

if __name__ == "__main__":
    main()

    # Sample command using language-specific writing prompts
    # python script.py --prompt_template prompt_files/lang_write/prompt_with_specs.txt --lang_prompts --dataset_id UniversalCEFR/universalcefr_test --model_id google/gemma-7b-it

    # Sample command using English writing prompts
    # python script.py --prompt_template prompt_files/lang_write/prompt_with_specs_en.txt --dataset_id UniversalCEFR/universalcefr_test --model_id google/gemma-7b-it
