import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
import wandb
from accelerate import init_empty_weights
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from tqdm import trange
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, set_seed
from transformers.modeling_utils import no_init_weights
from copy import deepcopy
from apply_qwen2_with_fixed_token_embedding import apply_qwen2_with_fixed_token_embedding_forward


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class ModelArgs:
    model_path: str = "Qwen/Qwen2.5-7B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"
    num_labels: int = 3

@dataclass
class DataArgs:
    train_data: str = "" #"RLHFlow/Mistral-ORM-Data"
    prm_version: str = ""

@dataclass
class OptimArgs:
    lr: float = 1e-5
    micro_batch_size: int = 1
    global_batch_size: Optional[int] = None
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    warmup_ratio: Optional[float] = 0.0


@dataclass
class TrainArgs:
    output_dir: str = "output"
    seed: int = 42
    wandb_project: str = "RewardModel"
    wandb_name: Optional[str] = None
    model: "ModelArgs" = field(default_factory=ModelArgs)
    data: "DataArgs" = field(default_factory=DataArgs)
    optim: "OptimArgs" = field(default_factory=OptimArgs)


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
GLOBAL_RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
IGNORE_INDEX = -100

PRM_LABEL_MAP = {
    '<|prm_negative|>': 0,
    '<|prm_neutral|>': 1,
    '<|prm_positive|>': 2,
}

"""
Dataset({
    features: ['query', 'response', 'prm_step_labels'],
    num_rows: 904524
})
"""

class RewardData(Dataset):
    def __init__(self, data_path: str, tokenizer: "PreTrainedTokenizer", prm_version: str):
        self._data = load_dataset('json', data_files=data_path, split="train")
        self._tokenizer = tokenizer
        
        self._label_placeholder_id = self._tokenizer('<|label_placeholder|>')['input_ids'][0]
        self._four_newline_id = self._tokenizer('\n\n\n\n')['input_ids'][0]
        self._prm_version = prm_version

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, "torch.Tensor"]:
        sample = self._data[index]
        
        question = sample["query"]
        answer = sample["response"]
        prm_step_labels = sample["prm_step_labels"]
        
        input_ids = self._tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ],
            tokenize=True,
            add_generation_prompt=False,
        )
        # print(f"\n\n\n\ninput_ids: {input_ids}\n\n\n\n")
        # input_ids = self._tokenizer.encode(input_ids, add_special_tokens=False)
        
        all_input_ids_seperate_token_num = 0
        for id in input_ids:
            if id == self._label_placeholder_id:
                all_input_ids_seperate_token_num += 1
        
        labels = deepcopy(input_ids)
        cur_seperate_token_num_in_labels = 0
        cur_seperate_token_id_in_labels = []
        
        def _convert_prm_label(prm_label: str) -> int:
            if prm_label in PRM_LABEL_MAP:
                return PRM_LABEL_MAP[prm_label]
            else:
                raise ValueError(f'Invalid prm label: {prm_label}')
        
        for idx, id in enumerate(labels):
            if id == self._label_placeholder_id:
                cur_seperate_token_num_in_labels += 1
                if self._prm_version == 'v2' or self._prm_version == 'v2_place_holder':
                    labels[idx] = _convert_prm_label(prm_step_labels[cur_seperate_token_num_in_labels - 1])
                    cur_seperate_token_id_in_labels.append(idx)
                elif self._prm_version == 'v6':
                    if cur_seperate_token_num_in_labels == all_input_ids_seperate_token_num:
                        labels[idx] = _convert_prm_label(prm_step_labels[0])
                        cur_seperate_token_id_in_labels.append(idx)
                    else:
                        labels[idx] = IGNORE_INDEX
                elif self._prm_version == 'v3':
                    if cur_seperate_token_num_in_labels == all_input_ids_seperate_token_num:
                        labels[idx] = _convert_prm_label(prm_step_labels[0])
                        cur_seperate_token_id_in_labels.append(idx)
                    else:
                        labels[idx] = IGNORE_INDEX
                else:
                    raise ValueError(f'Invalid prm version: {self._prm_version}')
            else:
                labels[idx] = IGNORE_INDEX
        
        cur_seperate_token_num_in_input_ids = 0
        cur_seperate_token_id_in_input_ids = []
        for idx, id in enumerate(input_ids):
            if id == self._label_placeholder_id:
                cur_seperate_token_num_in_input_ids += 1
                if self._prm_version == 'v2':
                    input_ids[idx] = self._four_newline_id
                    cur_seperate_token_id_in_input_ids.append(idx)
                elif self._prm_version == 'v2_place_holder':
                    cur_seperate_token_id_in_input_ids.append(idx)
                elif self._prm_version == 'v6':
                    input_ids[idx] = self._four_newline_id
                    if cur_seperate_token_num_in_input_ids == all_input_ids_seperate_token_num:
                        cur_seperate_token_id_in_input_ids.append(idx)
                elif self._prm_version == 'v3':
                    input_ids[idx] = self._four_newline_id
                    if cur_seperate_token_num_in_input_ids == all_input_ids_seperate_token_num:
                        input_ids[idx] = self._tokenizer('<|label_placeholder|>')['input_ids'][0]
                        cur_seperate_token_id_in_input_ids.append(idx)
                else:
                    raise ValueError(f'Invalid prm version: {self._prm_version}')
        
        attention_mask = [1] * len(input_ids)
        
        # print_rank0(f'{cur_seperate_token_id_in_input_ids=}')
        # print_rank0(f'{cur_seperate_token_id_in_labels=}')
        # print_rank0(f'{all_input_ids_seperate_token_num=}')
        # print_rank0(f'{self._prm_version=}')
        
        tokenized_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        tokenized_input = {k: torch.tensor(v) for k, v in tokenized_input.items()}
        return tokenized_input


def pad_feature(
    batch: Dict[str, List[Any]], feature: Dict[str, "torch.Tensor"], key: str, max_seq_len: int, pad_value: int = 0
) -> None:
    input_tensor = feature[key]
    cur_seq_len = input_tensor.size(-1)
    pad_size = max_seq_len - cur_seq_len
    if pad_size == 0:
        batch[key].append(input_tensor)
    else:
        pad_shape = list(input_tensor.shape)
        pad_shape[-1] = pad_size
        pad_tensor = torch.full(pad_shape, pad_value, dtype=input_tensor.dtype, device=input_tensor.device)
        batch[key].append(torch.cat((input_tensor, pad_tensor), dim=-1))


@dataclass
class DataCollatorWithPadding:
    num_micro_batches: int

    def collate(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)
        max_seq_len = max(feature["input_ids"].size(-1) for feature in features)
        for feature in features:
            pad_feature(batch, feature, "input_ids", pad_value=0, max_seq_len=max_seq_len)
            pad_feature(batch, feature, "attention_mask", pad_value=0, max_seq_len=max_seq_len)
            pad_feature(batch, feature, "labels", pad_value=IGNORE_INDEX, max_seq_len=max_seq_len)

        for input_name in batch.keys():
            batch[input_name] = torch.stack(batch[input_name], dim=0)

        return batch

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        micro_batch_size = len(features) // self.num_micro_batches
        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            micro_batches.append(self.collate(features[i : i + micro_batch_size]))

        return micro_batches


def print_rank0(*args):
    if LOCAL_RANK == 0:
        print(*args)


def create_init_fn(model: "nn.Module", device: Union[str, "torch.device"]) -> Callable[["nn.Module"], None]:
    param_occurrence = defaultdict(int)
    for _, param in model.named_parameters(remove_duplicate=False):
        param_occurrence[param] += 1

    duplicated_params = {param for param in param_occurrence.keys() if param_occurrence[param] > 1}
    materialized_params = {}

    def init_fn(module: "nn.Module"):
        for name, param in module.named_parameters(recurse=False):
            if param in duplicated_params:
                module._parameters[name] = materialized_params.setdefault(
                    param, nn.Parameter(torch.empty_like(param.data, device=device), requires_grad=param.requires_grad)
                )
            else:
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param.data, device=device), requires_grad=param.requires_grad
                )

    return init_fn


def train(args: TrainArgs):
    if args.data.prm_version == 'v6':
        apply_qwen2_with_fixed_token_embedding_forward()
    
    if args.optim.global_batch_size is None:
        args.optim.global_batch_size = args.optim.micro_batch_size * WORLD_SIZE
        gradient_accumulation_steps = 1
    else:
        if args.optim.global_batch_size % (args.optim.micro_batch_size * WORLD_SIZE) != 0:
            raise ValueError("`global_batch_size` must be divisible by `micro_batch_size` * `world_size`.")

        gradient_accumulation_steps = args.optim.global_batch_size // (args.optim.micro_batch_size * WORLD_SIZE)
        print_rank0(f"Use gradient accumulation: {gradient_accumulation_steps}.")

    print(f"Process rank: {GLOBAL_RANK}, world size: {WORLD_SIZE}")
    print_rank0(json.dumps(asdict(args), indent=2))

    set_seed(args.seed)
    torch.cuda.set_device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{LOCAL_RANK}"))

    if LOCAL_RANK == 0:
        datasets.logging.set_verbosity_info()
        transformers.logging.set_verbosity_info()
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()
    else:
        datasets.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(args.model.model_path)
    specific_tokens = ['<|label_placeholder|>']
    num_added_toks = tokenizer.add_tokens(
        specific_tokens
    )
    print_rank0(f"\n\n*****************\nWe have added {num_added_toks} tokens\n************************\n\n")
    
    train_dataset = RewardData(data_path=args.data.train_data, tokenizer=tokenizer, prm_version=args.data.prm_version)
    sampler = StatefulDistributedSampler(
        train_dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True, seed=args.seed
    )
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=args.optim.global_batch_size // WORLD_SIZE,
        sampler=sampler,
        num_workers=8,
        collate_fn=DataCollatorWithPadding(num_micro_batches=gradient_accumulation_steps),
        pin_memory=True,
        drop_last=True,
    )

    if GLOBAL_RANK == 0:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model.model_path,
            num_labels=args.model.num_labels,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    else:
        with no_init_weights(), init_empty_weights():
            config = AutoConfig.from_pretrained(args.model.model_path, num_labels=args.model.num_labels)
            model = AutoModelForTokenClassification.from_config(
                config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

    assert isinstance(model, PreTrainedModel)  # lint
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    wrap_policy = partial(
        lambda_auto_wrap_policy, lambda_fn=lambda module: module.__class__.__name__ in model._no_split_modules
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    model = model.float()  # enable mixed precision training
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,  # true if has freeze params
        sync_module_states=True,
        param_init_fn=create_init_fn(model, device="cuda") if GLOBAL_RANK != 0 else None,
    )

    train_steps = args.optim.max_steps if args.optim.max_steps else len(train_dataloader)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, args.optim.lr)
    # lr_scheduler = CosineAnnealingLR(optimizer, args.optim.num_train_epochs * train_steps,)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.optim.warmup_ratio * train_steps * args.optim.num_train_epochs, 
        num_training_steps=train_steps * args.optim.num_train_epochs,
    )
    loss_func = torch.nn.CrossEntropyLoss()

    if GLOBAL_RANK == 0:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=asdict(args))

    start_epoch, start_step, global_step = 0, 0, 0
    fsdp_model.train()
    print_rank0("Start training")
    for epoch in range(start_epoch, args.optim.num_train_epochs):
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        data_loader_tqdm = trange(
            train_steps,
            desc=f"Epoch {epoch + 1}/{args.optim.num_train_epochs}",
            initial=start_step,
            disable=LOCAL_RANK != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, train_steps):
            global_step += 1
            micro_batches: List[Dict[str, "torch.Tensor"]] = next(data_iterator)

            # if global_step == 1:
            #     for key, value in micro_batches[0].items():
            #         print(f"[rank {GLOBAL_RANK}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")

            total_loss = 0
            n_correct, n_total = 0, 0
            for micro_batch in micro_batches:
                micro_batch = {k: v.cuda(non_blocking=True) for k, v in micro_batch.items()}
                labels = micro_batch.pop("labels")
                logits: "torch.Tensor" = fsdp_model(**micro_batch, use_cache=False).logits
                logits = logits.view(-1, model.config.num_labels).float()
                labels = labels.view(-1)
                loss = loss_func(logits, labels) / len(micro_batches)
                loss.backward()
                total_loss += loss.item()

                n_correct += torch.where(labels != IGNORE_INDEX, logits.argmax(dim=-1) == labels, 0).sum().item()
                n_total += (labels != IGNORE_INDEX).sum().item()

            grad_norm = fsdp_model.clip_grad_norm_(args.optim.max_grad_norm).item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            metrics = torch.tensor([total_loss, grad_norm, n_correct, n_total], device="cuda")
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, grad_norm = metrics[0].item() / WORLD_SIZE, metrics[1].item() / WORLD_SIZE
            n_correct, n_total = metrics[2].item(), metrics[3].item()
            accuracy = n_correct / n_total
            lr = max(lr_scheduler.get_last_lr())
            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, accuracy: {accuracy:.2f}"
            )
            data_loader_tqdm.update()

            if GLOBAL_RANK == 0:
                train_metrics = {
                    "training/loss": total_loss,
                    "training/grad_norm": grad_norm,
                    "training/lr": lr,
                    "training/accuracy": accuracy,
                }
                wandb.log(train_metrics, step=global_step)

        data_loader_tqdm.close()

    torch.cuda.synchronize()
    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config):
        state_dict = fsdp_model.state_dict()

    if GLOBAL_RANK == 0:
        os.makedirs(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir, state_dict=state_dict)

    dist.barrier()
    dist.destroy_process_group()


def main():
    # Get CLI arguments first (highest priority)
    cli_args = OmegaConf.from_cli()
    
    # Load yaml config if provided (second priority)
    config_file = cli_args.pop("config", None)
    if config_file:
        yaml_args = OmegaConf.load(config_file)
    else:
        yaml_args = OmegaConf.create({})
        
    # Get default arguments (lowest priority)
    default_args = OmegaConf.structured(TrainArgs)
    
    # Merge with priority: CLI > YAML > defaults
    train_args = OmegaConf.merge(default_args, yaml_args, cli_args)
    
    train(OmegaConf.to_object(train_args))


if __name__ == "__main__":
    main()

# s1 \n\n\n\n s2 \n\n\n\n s3 \n\n\n\n ... sn \n\n\n\n
# s1 <label_placeholder> s2 <label_placeholder> s3 <label_placeholder> ... sn <label_placeholder>
# s1 \n\n\n\n s2 \n\n\n\n s3 \n\n\n\n ... sn <label_placeholder>