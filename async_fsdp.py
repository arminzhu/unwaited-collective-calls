"""
该版本使用了backward_hook,这样会出现unwaited collective的bug
"""

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard, Replicate, Partial, distribute_tensor, zeros
import torch.distributed._functional_collectives as funcol
import blocks
import math

layer_types = [torch.nn.Linear, ]
types_1 = [torch.nn.Embedding, blocks.FinalBlock, ]
types_4 = [blocks.TransformerBlock, ]
types = [torch.nn.Embedding, blocks.TransformerBlock, blocks.FinalBlock]

class L_F(torch.autograd.Function):
    """
    act-ckpt and update parameters in backward, for a Linear model demo
    """
    UPDATES_COUNT = 0
    REDUCE_SCATTER_COUNT = 0

    param_info = None
    rank = None
    mesh = None
    device = None
    reduce_op = dist.ReduceOp.SUM

    param_idxs = dict()  # dict[id:(int,int,str)]
    max_inner_idx = []  # list[int]

    total_block_nums = 0
    current_block = -1
    max_act_ckpt_idx = -1
    cache_size = 1

    embedding_padding_idx = None
    weight_cache = None
    modules = torch.nn.ModuleList()  # Assigned in forward
    model_parameters = []  # the ptr of model param, equal to model.param, 32bits

    full_parameters = []
    full_param_size = []
    sharded_grad = []
    sharded_parameters = []  # 32bits
    sharded_m = []
    sharded_v = []

    inputs_data = []  # a list of intermediate inputs data for each block_num. Assigned in forward
    outputs_data = []  # a list of intermediate outputs data for each block_num
    cuda_state = []  # a list of cuda_state for each block_num. Assigned in forward

    states_H2D_event = []

    grad_reduce_scatter_event = []
    param_all_gather_event = []

    start_pos = None
    freqs_cis = None
    mask = None

    lr = 1e-3
    alpha = 0.9
    beta = 0.9
    eps = 1e-8
    weight_decay = 1e-2
    T = 0
    lr_fixed = None

    dtype_bit16 = torch.bfloat16
    dtype_bit32 = torch.float32

    full_initialized = False
    is_reset = True
    stream_H2D = None
    stream_D2H = None
    stream_collective = None

    @staticmethod
    def initialize(param_info, gpu_mesh, cpu_mesh, rank=0, device="cpu",
                   lr=1e-3, alpha=0.9, beta=0.99, eps=1e-8, weight_decay=1e-2, ):
        L_F.param_info = param_info
        L_F.rank = rank
        L_F.mesh = gpu_mesh
        L_F.device = device
        L_F.current_block = -1
        L_F.lr = lr
        L_F.alpha = alpha
        L_F.beta = beta
        L_F.eps = eps
        L_F.weight_decay = weight_decay
        L_F.stream_H2D = torch.cuda.Stream(device)
        L_F.stream_D2H = torch.cuda.Stream(device)
        L_F.stream_collective = torch.cuda.Stream(device)

    @staticmethod
    def param_reinitialize(param_info, dtype, device):
        with torch.no_grad():
            param_data = None
            if param_info.module_type == torch.nn.modules.sparse.Embedding:
                param_data = torch.empty(param_info.param_size, dtype=dtype, device=device, requires_grad=True)
                if param_info.name == "weight":
                    torch.nn.init.normal_(param_data)
                    if L_F.embedding_padding_idx is not None:
                        param_data[L_F.embedding_padding_idx].fill_(0)
                    L_F.weight_cache = param_data
                else:
                    raise TypeError("param_info.module_type is Embedding, it only has weight!")
            elif param_info.module_type == torch.nn.modules.normalization.LayerNorm:
                param_data = torch.empty(param_info.param_size, dtype=dtype, device=device, requires_grad=True)
                if param_info.name == "weight":
                    torch.nn.init.ones_(param_data)
                    L_F.weight_cache = param_data
                elif param_info.name == "bias":
                    torch.nn.init.zeros_(param_data)
                else:
                    raise ValueError(f"Unknown module name: {param_info.name}!")
            elif param_info.module_type == torch.nn.modules.linear.Linear:
                param_data = torch.empty(param_info.param_size, dtype=dtype, device=device, requires_grad=True)
                if param_info.name == "weight":
                    param_data = torch.nn.init.kaiming_uniform_(param_data, a=math.sqrt(5.))
                    L_F.weight_cache = param_data
                elif param_info.name == "bias":
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(L_F.weight_cache)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    torch.nn.init.uniform_(param_data, -bound, bound)
                else:
                    raise ValueError(f"Unknown module name: {param_info.name}!")
            elif param_info.module_type == blocks.Conv1D:
                if param_info.name == "weight":
                    param_data = torch.empty(param_info.param_size, dtype=dtype, device=device, requires_grad=True)
                    torch.nn.init.normal_(param_data, std=0.02)
                    L_F.weight_cache = param_data
                elif param_info.name == "bias":
                    param_data = torch.zeros(param_info.param_size, dtype=dtype, device=device, requires_grad=True)
                else:
                    raise ValueError(f"Unknown module name: {param_info.name}!")
            else:
                raise TypeError("Unknown module type!")
            return param_data

    @staticmethod
    def add_an_empty_state():
        L_F.inputs_data.append(None)
        L_F.outputs_data.append(None)
        L_F.cuda_state.append(None)
        L_F.model_parameters.append(dict())
        L_F.full_param_size.append(dict())
        L_F.sharded_grad.append(dict())
        L_F.sharded_parameters.append(dict())
        L_F.full_parameters.append(dict())
        L_F.sharded_m.append(dict())
        L_F.sharded_v.append(dict())
        L_F.states_H2D_event.append(dict())
        L_F.grad_reduce_scatter_event.append(dict())
        L_F.param_all_gather_event.append(None)

    @staticmethod
    def get_leaf_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
        """Return the leaf modules of the given model"""
        with torch.no_grad():
            return [m for m in model.modules() if not list(m.children())]

    @staticmethod
    def register_grad_reduce_scatter_hook(module, hook):
        with torch.no_grad():
            leaf_modules = L_F.get_leaf_modules(module)
            for m in leaf_modules:
                m.register_full_backward_hook(hook)

    @staticmethod
    def creat_states_impl(module, block_idx):
        inner_idx = -1
        for name, param in module.named_parameters():
            inner_idx += 1
            param.data = L_F.param_reinitialize(L_F.param_info[id(param)],
                                                dtype=L_F.dtype_bit32,
                                                device=L_F.device)
            L_F.model_parameters[-1][name] = param
            L_F.full_param_size[-1][name] = param.data.numel() * param.data.element_size()
            L_F.sharded_parameters[-1][name] = distribute_tensor(param.data,
                                                                 L_F.mesh,
                                                                 [Shard(dim=0), ])
            _shape = L_F.sharded_parameters[-1][name].to_local().shape
            L_F.sharded_m[-1][name] = torch.full(_shape,
                                                 fill_value=0.,
                                                 dtype=L_F.dtype_bit32,
                                                 device=L_F.device,
                                                 requires_grad=False)

            L_F.sharded_v[-1][name] = torch.full(_shape,
                                                 fill_value=0.,
                                                 dtype=L_F.dtype_bit32,
                                                 device=L_F.device,
                                                 requires_grad=False)

            L_F.full_parameters[-1][name] = L_F.sharded_parameters[-1][name].redistribute(
                device_mesh=L_F.mesh, placements=[Replicate(), ], async_op=False)
            L_F.param_all_gather_event[block_idx] = torch.cuda.Event()
            L_F.param_all_gather_event[block_idx].record()
            L_F.param_idxs[id(param)] = (block_idx, inner_idx, name)
        L_F.max_inner_idx.append(inner_idx)

    @staticmethod
    def creat_state_and_all_gather(module, block_idx):
        L_F.add_an_empty_state()
        L_F.modules.append(module)
        with torch.no_grad():
            L_F.creat_states_impl(module, block_idx)
            # If we 'register_grad_reduce_scatter_hook' on the block 0, it also doesn't work.
            if block_idx != 0:
                L_F.register_grad_reduce_scatter_hook(module, L_F.module_backward_hook)
        L_F.total_block_nums += 1

    @staticmethod
    def reset():
        # print(f"UPDATES_COUNT is {L_F.UPDATES_COUNT}, REDUCE_SCATTER_COUNT is {L_F.REDUCE_SCATTER_COUNT}")
        L_F.T += 1
        L_F.current_block = -1
        temp_fixed_lr = L_F.lr * math.sqrt(1. - L_F.beta ** L_F.T) / (
                1. - L_F.alpha ** L_F.T)
        L_F.lr_fixed = torch.tensor(temp_fixed_lr, dtype=L_F.dtype_bit32)

    @staticmethod
    def get_local_element(T):
        with torch.no_grad():
            with funcol.allow_inflight_collective_as_graph_input_ctx():
                if type(T) == torch.Tensor:
                    return T
                elif type(T) == torch.distributed.tensor.DTensor:
                    _local = T.to_local()
                    if isinstance(_local, funcol.AsyncCollectiveTensor):
                        # funcol.wait_tensor(_local.elem)
                        _local.wait()
                        return _local.wait()
                    else:
                        return _local
                else:
                    raise TypeError("The data type of 'T' is not supported!")

    @staticmethod
    def all_gather_param(block_idx):
        with torch.no_grad():
            with torch.cuda.stream(L_F.stream_collective):
                for name, param in L_F.modules[block_idx].named_parameters():
                    L_F.full_parameters[block_idx][name] = L_F.sharded_parameters[block_idx][name].redistribute(
                        device_mesh=L_F.mesh, placements=[Replicate(), ], async_op=True)
                L_F.param_all_gather_event[block_idx] = torch.cuda.Event()
                L_F.param_all_gather_event[block_idx].record()

    @staticmethod
    def prepare_parameter(block_idx):
        with torch.no_grad():
            torch.cuda.current_stream().wait_event(L_F.param_all_gather_event[block_idx])
            for name, param in L_F.modules[block_idx].named_parameters():
                param.data.untyped_storage().resize_(L_F.full_param_size[block_idx][name]).copy_(
                    L_F.get_local_element(L_F.full_parameters[block_idx][name]).untyped_storage())
                L_F.full_parameters[block_idx][name] = None
                L_F.model_parameters[block_idx][name].data.requires_grad = True

    @staticmethod
    def shard_param(block_idx):
        with torch.no_grad():
            for name, param in L_F.modules[block_idx].named_parameters():
                param.grad = None
                param.data.untyped_storage().resize_(0)

    @staticmethod
    def module_backward_hook(module, grad_input, grad_output):
        """
        The 'module_backward_hook' can do 'reduce_scatter_grad' on blocks 1,2,3,...;
         but it not work on the block 0.
        """
        with torch.no_grad():
            with torch.cuda.stream(L_F.stream_collective):
                L_F.stream_collective.wait_stream(torch.cuda.default_stream(L_F.device))
                for name, param in module.named_parameters():
                    _block_idx, _inner_idx, _name = L_F.param_idxs[id(param)]
                    if param.grad is not None:
                        L_F.sharded_grad[_block_idx][_name] = DTensor.from_local(param.grad, L_F.mesh, placements=[
                            Partial(reduce_op='sum')]).redistribute(device_mesh=L_F.mesh, placements=[Shard(dim=0), ],
                                                                    async_op=True)
                        param.grad.record_stream(L_F.stream_collective)
                        param.grad = None
                        L_F.REDUCE_SCATTER_COUNT += 1
                        L_F.grad_reduce_scatter_event[_block_idx][_name] = torch.cuda.Event()
                        L_F.grad_reduce_scatter_event[_block_idx][_name].record()
                    else:
                        print(f"module_backward_hook None block_idx is {_block_idx}")

    @staticmethod
    def reduce_scatter_grad(block_idx):
        """
        Only works on the block_idx=0, and runs in synchronous mode.
        :param block_idx: block_idx is 0.
        :return:
        """
        with torch.no_grad():
            with torch.cuda.stream(L_F.stream_collective):
                for name, param in L_F.modules[block_idx].named_parameters():
                    if param.grad is not None:
                        L_F.sharded_grad[block_idx][name] = DTensor.from_local(
                            param.grad, L_F.mesh, placements=[Partial(reduce_op='sum')]).redistribute(
                            device_mesh=L_F.mesh, placements=[Shard(dim=0), ], async_op=False)
                        param.grad.record_stream(L_F.stream_collective)
                        param.grad = None
                        L_F.REDUCE_SCATTER_COUNT += 1
                        L_F.grad_reduce_scatter_event[block_idx][name] = torch.cuda.Event()
                        L_F.grad_reduce_scatter_event[block_idx][name].record()
                    else:
                        print("reduce_scatter_grad 'param.grad' is None")
    
    @staticmethod
    def update_parameters(block_idx):
        # torch.cuda.synchronize()
        with torch.no_grad():
            for name, param in L_F.sharded_parameters[block_idx].items():
                torch.cuda.current_stream(L_F.device).wait_event(L_F.grad_reduce_scatter_event[block_idx][name])
                p = L_F.get_local_element(L_F.sharded_parameters[block_idx][name])
                g = L_F.get_local_element(L_F.sharded_grad[block_idx][name])
                m = L_F.sharded_m[block_idx][name]
                v = L_F.sharded_v[block_idx][name]
                adamw_(p, g, m, v,
                       L_F.lr_fixed,
                       L_F.lr * L_F.weight_decay,
                       L_F.alpha,
                       L_F.beta,
                       L_F.eps)
                g.record_stream(torch.cuda.current_stream(L_F.device))
                L_F.sharded_grad[block_idx][name] = None
                del g
                L_F.UPDATES_COUNT += 1
    
    @staticmethod
    def save_state(cuda_state, block):
        """
        save cuda_state for dropout, etc.
        """
        L_F.cuda_state[block] = cuda_state

    @staticmethod
    def set_state(block):
        """
        set cuda_state for dropout, etc.
        """
        torch.cuda.set_rng_state(L_F.cuda_state[block], device=L_F.device)

    @staticmethod
    def save_input_data(data, block):
        L_F.inputs_data[block] = list(data)

    @staticmethod
    def forward_run_module(run_module, block_idx, *args):
        _inputs = None
        if type(run_module) == torch.nn.Embedding:
            _inputs = (args[0].to(dtype=torch.long, device=args[0].device),)
        elif type(run_module) in types:
            _inputs = args
        else:
            raise TypeError("Unknown module type!")
        L_F.save_input_data(_inputs, block_idx)
        L_F.save_state(torch.cuda.get_rng_state(device=L_F.device), block_idx)
        L_F.prepare_parameter(block_idx)
        if type(run_module) in types:
            with torch.no_grad():
                _outputs = run_module(*_inputs)
                return _outputs
        else:
            raise TypeError("Unknown type!")

    @staticmethod
    def forward_with_init(run_module, block_idx, *args):
        if block_idx > 0:
            L_F.shard_param(block_idx - 1)
        L_F.creat_state_and_all_gather(run_module, block_idx)
        out = L_F.forward_run_module(run_module, block_idx, *args)
        return out

    @staticmethod
    def forward_pure(run_module, block_idx, *args):
        if block_idx == 0:
            L_F.all_gather_param(block_idx)
        if block_idx != L_F.total_block_nums - 1:
            out = L_F.forward_run_module(run_module, block_idx, *args)
            L_F.all_gather_param(block_idx + 1)
            L_F.shard_param(block_idx)
            return out
        elif block_idx == L_F.total_block_nums - 1:
            out = L_F.forward_run_module(run_module, block_idx, *args)
            return out
        else:
            raise ValueError(f"Block_idx should in [0, {L_F.total_block_nums - 1}], but got {block_idx}!")

    @staticmethod
    def forward(ctx, run_module, *args):
        torch.cuda.default_stream().synchronize()
        L_F.current_block += 1
        block_idx = L_F.current_block
        ctx.block_idx = block_idx
        if L_F.is_reset:
            L_F.is_reset = False
        if not L_F.full_initialized and block_idx > len(L_F.modules) - 1:
            out = L_F.forward_with_init(run_module, block_idx, *args)
            return out
        else:
            out = L_F.forward_pure(run_module, block_idx, *args)
            return out

    @staticmethod
    def get_input_data(block_idx):
        L_F.inputs_data[block_idx][0] = L_F.inputs_data[block_idx][0].detach()
        f = L_F.modules[block_idx]
        if type(f) != torch.nn.Embedding:
            L_F.inputs_data[block_idx][0].requires_grad = True
        else:
            L_F.inputs_data[block_idx][0] = L_F.inputs_data[block_idx][0].to(dtype=torch.long)
        return L_F.inputs_data[block_idx]

    @staticmethod
    def recompute(block_idx):
        if block_idx != L_F.total_block_nums - 1:
            L_F.prepare_parameter(block_idx)
        torch.cuda.set_rng_state(L_F.cuda_state[block_idx])
        f = L_F.modules[block_idx]
        with torch.enable_grad():
            if type(f) in types:
                return f(*L_F.inputs_data[block_idx])
            else:
                raise TypeError("Unknown function type!")


    @staticmethod
    def get_inputs_grads(inputs_data):
        inputs_grads = []
        for element in inputs_data:
            if hasattr(element, 'grad'):
                inputs_grads.append(element.grad)
            else:
                inputs_grads.append(None)
        return tuple(inputs_grads)

    @staticmethod
    def backward_impl(block_idx, *args):
        if block_idx != 0:
            L_F.all_gather_param(block_idx - 1)
        inputs = L_F.get_input_data(block_idx)
        L_F.set_state(block_idx)
        outputs = L_F.recompute(block_idx)
        torch.autograd.backward(outputs, args)
        if block_idx == 0:
            L_F.reduce_scatter_grad(block_idx)
        L_F.update_parameters(block_idx)
        inputs_grads = L_F.get_inputs_grads(inputs)
        L_F.shard_param(block_idx)
        return inputs_grads

    @staticmethod
    def backward(ctx, *args):
        torch.cuda.default_stream().synchronize()
        if not L_F.full_initialized:
            L_F.full_initialized = True
        if not L_F.is_reset:
            L_F.reset()
            L_F.is_reset = True
        inputs_grads = L_F.backward_impl(ctx.block_idx, *args)
        f = L_F.modules[ctx.block_idx]
        if type(f) in types:
            return (None,) + inputs_grads
        else:
            raise TypeError("Unknown function type!")


def Async_FSDP(run_function, *args, **kwargs):
    """

    :param block_num:
    :param run_function: nn.ModuleList
    :param args: input->tensor
    :param kwargs:
    :return:
    """
    return L_F.apply(run_function, *args)


# @torch.compile(fullgraph=True, dynamic=True, options={"epilogue_fusion": True, "max_autotune": True})
def adamw_(p, g, m, v, lr, weight_decay=1e-2, alpha=0.9, beta=0.99, eps=1e-8):
    m.mul_(alpha).add_(g, alpha=1. - alpha)
    v.mul_(beta).addcmul_(g, g, value=1. - beta)
    p.data.mul_(1 - weight_decay).addcdiv_(m, v.sqrt().add_(eps), value=-lr)
