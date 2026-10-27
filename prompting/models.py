"""Lazy Hugging Face model loading and the six original generation paths."""

from __future__ import annotations

import os

SYSTEM_PROMPT = "You are a helpful assistant for CEFR classification."


def model_family(model_id):
    name = model_id.lower()
    for fragment, family in (("gemma-3-12b-it", "gemma3"), ("eurollm-9b-instruct", "eurollm"),
                             ("gemma-7b-it", "gemma"), ("aya-101", "aya"),
                             ("llama-3.1-8b", "llama"), ("bloomz", "bloomz")):
        if fragment in name:
            return family
    raise ValueError(f"unsupported model {model_id!r}; select --model-family for a compatible local model")


def generation_options(family):
    """Keep generation controls from the original script, including sampling."""
    if family in ("gemma3", "llama"):
        return {"max_new_tokens": 10, "do_sample": False}
    if family == "aya":
        return {"do_sample": True, "min_length": 1, "max_length": 100}
    if family == "bloomz":
        return {"max_new_tokens": 128}
    if family in ("eurollm", "gemma"):
        return {"max_new_tokens": 10}
    raise ValueError(f"unsupported model family {family!r}")


class Predictor:
    def __init__(self, model, tokenizer, family, torch_module):
        self.model = model
        self.tokenizer = tokenizer
        self.family = family
        self.torch = torch_module

    def __call__(self, prompt):
        model, tokenizer = self.model, self.tokenizer
        device = model.get_input_embeddings().weight.device
        if str(device) == "meta":
            device = model.device
        if str(device) == "meta":
            raise ValueError("model input device is unresolved after offloading; use --device cpu "
                             "or a configuration with input embeddings on CPU/CUDA")
        options = generation_options(self.family)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        with self.torch.inference_mode():
            if self.family == "gemma3":
                messages = [{"role": row["role"], "content": [{"type": "text", "text": row["content"]}]}
                            for row in messages]
                inputs = tokenizer.apply_chat_template(
                    messages, padding="longest", pad_to_multiple_of=8, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt")
                inputs = {key: value.contiguous().to(device) for key, value in inputs.items()}
                output = model.generate(**inputs, **options)[0][inputs["input_ids"].shape[-1]:]
            elif self.family in ("llama", "eurollm"):
                inputs = tokenizer.apply_chat_template(messages, tokenize=True,
                                                       add_generation_prompt=True, return_tensors="pt").to(device)
                if self.family == "llama":
                    options["eos_token_id"] = [tokenizer.eos_token_id,
                                               tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                output = model.generate(inputs, **options)[0][inputs.shape[-1]:]
            else:
                tokenize_options = {"return_tensors": "pt"}
                if self.family == "aya":
                    tokenize_options.update(padding=True, truncation=True, max_length=128)
                inputs = tokenizer(prompt, **tokenize_options).to(device)
                output = model.generate(**inputs, **options)[0]
                if self.family != "aya":
                    output = output[inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(output, skip_special_tokens=True).strip()


def load_predictor(args):
    """Load local inference resources. Credentials use HF_TOKEN or saved Hub login."""
    family = args.model_family or model_family(args.model_id)
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                              AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration,
                              set_seed)

    if args.seed is not None:
        set_seed(args.seed)
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise ValueError("--device cuda requires CUDA; use --device cpu or auto")
    common = {"revision": args.revision, "cache_dir": str(args.cache_dir) if args.cache_dir else None,
              "local_files_only": args.local_files_only, "trust_remote_code": args.trust_remote_code}
    loading = {**common, "device_map": "auto" if use_cuda else {"": "cpu"}}
    # Aya originally used the model's default dtype. Retain that on CUDA.
    if args.dtype != "auto":
        loading["torch_dtype"] = getattr(torch, args.dtype)
    elif not use_cuda:
        loading["torch_dtype"] = torch.float32
    elif family != "aya":
        loading["torch_dtype"] = torch.bfloat16
    model_class = (Gemma3ForConditionalGeneration if family == "gemma3" else
                   AutoModelForSeq2SeqLM if family == "aya" else AutoModelForCausalLM)
    model = model_class.from_pretrained(args.model_id, **loading)
    loader = AutoProcessor if family == "gemma3" else AutoTokenizer
    tokenizer = loader.from_pretrained(args.model_id, **common)
    if family == "gemma3":
        tokenizer.tokenizer.padding_side = "left"
    elif family == "aya":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
    model.eval()
    return Predictor(model, tokenizer, family, torch)
