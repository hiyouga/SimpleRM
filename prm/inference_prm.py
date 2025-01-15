from dataclasses import dataclass

import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoModelForTokenClassification, AutoTokenizer

PRM_LABEL_MAP = {
    '<|prm_negative|>': 0,
    '<|prm_neutral|>': 1,
    '<|prm_positive|>': 2,
}
PRM_LABEL_MAP_INV = {v: k for k, v in PRM_LABEL_MAP.items()}

@dataclass
class InferArgs:
    model_path: str = "output"
    seed: int = 42
    num_labels: int = 3
    prm_version: str = "v6"
    

question = "In 7 years, Kaylee will be 3 times as old as Matt is now. If Matt is currently 5 years old, how old is Kaylee now?"
answer = [
    "In 7 years, Kaylee will be 3 * 5 = <<3*5=15>>15 years old.",
    "Kaylee's current age is 15 - 7 = <<15-7=8>>8 years old. The answer is: 8"
]

def inference(args: InferArgs):
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=args.num_labels,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    
    answer_text = '<|label_placeholder|>'.join(answer) + '<|label_placeholder|>'
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_text}
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    label_placeholder_id = tokenizer('<|label_placeholder|>')['input_ids'][0]
    four_newline_id = tokenizer('\n\n\n\n')['input_ids'][0]
    
    label_index_list = []
    for idx, id in enumerate(input_ids):
        if id == label_placeholder_id:
            if args.prm_version == "v6":
                label_index_list.append(idx)
                input_ids[idx] = four_newline_id
            else:
                raise ValueError(f"Unsupported PRM version: {args.prm_version}")
    
    with torch.inference_mode():
        logits = model(torch.tensor([input_ids], device="cuda")).logits

    logits = logits[0, torch.tensor(label_index_list, device="cuda")]
    step_labels = torch.argmax(logits, dim=-1)
    step_labels = [PRM_LABEL_MAP_INV[label.cpu().item()] for label in step_labels]
    return step_labels

def main():
    cli_args = OmegaConf.from_cli()
    default_args = OmegaConf.structured(InferArgs)
    infer_args = OmegaConf.merge(default_args, cli_args)
    step_labels = inference(OmegaConf.to_object(infer_args))
    print(step_labels)


if __name__ == "__main__":
    main()