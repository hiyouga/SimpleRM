import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import datasets
import torch
import torch.distributed as dist
import transformers
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from tqdm import trange
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, set_seed


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class ModelArgs:
    model_path: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")


@dataclass
class DataArgs:
    train_data: str = field(default="RLHFlow/Mistral-ORM-Data")


@dataclass
class OptimArgs:
    lr: float = field(default=1e-5)
    micro_batch_size: int = field(default=1)
    global_batch_size: Optional[int] = field(default=None)
    num_train_epochs: int = field(default=1)
    max_grad_norm: float = field(default=1.0)


@dataclass
class TrainArgs:
    output_dir: str = field(default="output")
    seed: int = field(default=42)
    wandb_project: str = field(default="RewardModel")
    wandb_name: Optional[str] = field(default=None)
    model: "ModelArgs" = field(default_factory=ModelArgs)
    data: "DataArgs" = field(default_factory=DataArgs)
    optim: "OptimArgs" = field(default_factory=OptimArgs)


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
GLOBAL_RANK = int(os.getenv("RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
IGNORE_INDEX = -100


class RewardData(Dataset):
    def __init__(self, data_path: str, tokenizer: "PreTrainedTokenizer"):
        self._data = load_dataset(data_path, split="train")
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, "torch.Tensor"]:
        sample = self._data[index]
        question, answer = str(sample["conversations"][0]["content"]).split("Step 1", maxsplit=1)
        answer = "Step 1" + answer
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        )
        input_ids = self._tokenizer.encode(text, add_special_tokens=False)
        attention_mask = [1] * len(input_ids)
        label = 1 if sample["conversations"][1]["content"] == "+" else 0
        labels = [-100] * (len(input_ids) - 1) + [label]
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


def train(args: TrainArgs):
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
    train_dataset = RewardData(data_path=args.data.train_data, tokenizer=tokenizer)
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

    model = AutoModelForTokenClassification.from_pretrained(
        args.model.model_path,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
        low_cpu_mem_usage=True,
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
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
        use_orig_params=False,  # true if has freeze params
    )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, args.optim.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, args.optim.num_train_epochs * len(train_dataloader))
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
            len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{args.optim.num_train_epochs}",
            initial=start_step,
            disable=LOCAL_RANK != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, len(train_dataloader)):
            global_step += 1
            micro_batches: List[Dict[str, "torch.Tensor"]] = next(data_iterator)

            if global_step == 1:
                for key, value in micro_batches[0].items():
                    print(f"[rank {GLOBAL_RANK}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")

            total_loss = 0
            for micro_batch in micro_batches:
                micro_batch = {k: v.cuda(non_blocking=True) for k, v in micro_batch.items()}
                labels = micro_batch.pop("labels")
                logits: "torch.Tensor" = fsdp_model(**micro_batch, use_cache=False).logits
                logits = logits.view(-1, model.config.num_labels).float()
                labels = labels.view(-1)
                loss = loss_func(logits, labels) / len(micro_batches)
                loss.backward()
                total_loss += loss.item()

            grad_norm = fsdp_model.clip_grad_norm_(args.optim.max_grad_norm).item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            metrics = torch.tensor([total_loss, grad_norm], device="cuda")
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, grad_norm = metrics[0].item() / WORLD_SIZE, metrics[1].item() / WORLD_SIZE
            lr = max(lr_scheduler.get_last_lr())
            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if GLOBAL_RANK == 0:
                train_metrics = {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                wandb.log(train_metrics, step=global_step)

        data_loader_tqdm.close()

    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config):
        state_dict = model.state_dict()

    if GLOBAL_RANK == 0:
        os.makedirs(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(args.output_dir, state_dict=state_dict)

    dist.barrier()
    dist.destroy_process_group()


def main():
    cli_args = OmegaConf.from_cli()
    default_args = OmegaConf.structured(TrainArgs)
    train_args = OmegaConf.merge(default_args, cli_args)
    train(OmegaConf.to_object(train_args))


if __name__ == "__main__":
    main()
