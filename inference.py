from dataclasses import dataclass

import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class InferArgs:
    model_path: str = "output"
    seed: int = 42


def inference(args: InferArgs):
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    question = "In 7 years, Kaylee will be 3 times as old as Matt is now. If Matt is currently 5 years old, how old is Kaylee now?"
    answer = "Step 1: In 7 years, Kaylee will be 3 * 5 = <<3*5=15>>15 years old.\nStep 2: Kaylee's current age is 15 - 7 = <<15-7=8>>8 years old. The answer is: 8"
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    with torch.inference_mode():
        logits = model(torch.tensor([input_ids], device="cuda")).logits

    print(logits[0, -1])


def main():
    cli_args = OmegaConf.from_cli()
    default_args = OmegaConf.structured(InferArgs)
    infer_args = OmegaConf.merge(default_args, cli_args)
    inference(OmegaConf.to_object(infer_args))


if __name__ == "__main__":
    main()
