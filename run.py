import data
import argparse
import torch
from torch.distributed.device_mesh import init_device_mesh
import os
import trainer
os.environ["USE_LIBUV"] = "0"
import torch.distributed as dist
import torch.multiprocessing as mp
import copy


class ParamInfos:
    def __init__(self, param_id, name, module_type, param_size):
        self.param_id = param_id
        self.name = name
        self.module_type = module_type
        self.param_size = param_size


def get_leaf_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """返回模型的叶子层"""
    return [m for m in model.modules() if not list(m.children())]


# dict[str, str | int]
def get_model_info(model: torch.nn.Module) -> dict[int, ParamInfos]:
    leaf_modules = get_leaf_modules(model)
    model_states = dict()
    idx = 0
    for module in leaf_modules:
        for name, param in module.named_parameters():
            # print(name, param.shape)
            model_states[idx] = ParamInfos(id(param), name, type(module), param.shape)
            idx += 1
    return model_states


def resize_storage(model: torch.nn.Module):
    for name, param in model.named_parameters():
        # print(name, param.size())
        param.data = torch.empty(0)
        # param.data.untyped_storage().resize_(0)


def reset_model_info(new_model: torch.nn.Module, model_info: dict[int, ParamInfos]) -> dict[int, ParamInfos]:
    new_model_info = dict()
    leaf_modules = get_leaf_modules(new_model)
    idx = 0
    for module in leaf_modules:
        for name, param in module.named_parameters():
            if name == model_info[idx].name:
                model_info[idx].param_id = id(param)
                new_model_info[id(param)] = copy.deepcopy(model_info[idx])
            # ParamArgs(id(param), name, type(module), param.shape)
            idx += 1
    return new_model_info


def print_info(model_info: dict[int, ParamInfos]):
    for key, val in model_info.items():
        print(f"{key}: {val.param_id}, {val.name}, {val.module_type}, {val.param_size}")


def worker(rank, world_size, base_model, base_model_info):
    torch.cuda.set_device(f"cuda:{rank}")
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    GPU_mesh = init_device_mesh("cuda", (world_size,))
    CPU_mesh = None
    _device=torch.device(f"cuda:{rank}")
    tokenizer_path = "./gpt2"
    text_data = eval(f"data.text_128_{rank}")
    model = copy.deepcopy(base_model)
    model_info = copy.deepcopy(base_model_info)
    model_info = reset_model_info(model, model_info)
    # if rank == 1:
    #     print_info(base_model_info)
    #     print_info(model_info)
    tokenizer = trainer.build_tokenizer(tokenizer_path)
    model=model.cuda(device=_device)
    model.ombb_initialize(model_info, gpu_mesh=GPU_mesh, cpu_mesh=CPU_mesh, rank=rank, device=_device,
                          train_lr=0.0005, alpha=0.9, beta=0.95, eps=1e-8, weight_decay=1e-2, )

    MyTrainer=trainer.GPT2_Trainer(model, tokenizer)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    MyTrainer.text_train(text_data, device=_device, rank = rank)

    dist.destroy_process_group()


def main(world_size, model, model_info):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23456'
    mp.set_start_method('spawn')
    mp.spawn(worker, args=(world_size, model, model_info), nprocs=world_size, join=True)


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    args = parser.parse_args()
    config_path = "./params.json"
    model = trainer.build_model(
        config_path=config_path,
        max_seq_len=args.seqlen,
        max_batch_size=args.bsz,
        is_train=True,
        model_dtype=torch.bfloat16,
        dtype_bit16=torch.bfloat16,
        dtype_bit32=torch.float32,
    )
    model_info = get_model_info(model)
    resize_storage(model)

    main(2, model, model_info)
