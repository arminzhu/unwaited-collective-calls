import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    # embedding dim
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 4096


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dtype=torch.bfloat16, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        n_out (`int`): The number of output features.
        n_in (`int`): The number of input features.
    """

    def __init__(self, n_out, n_in,dtype=torch.bfloat16):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in
        self.weight = nn.Parameter(torch.empty(n_in, n_out,dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(n_out,dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(n_out={n_out}, n_in={n_in})".format(**self.__dict__)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.n_out,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_train=False,dtype=torch.bfloat16):
        super().__init__()
        self.is_train = is_train
        # n_kv_heads为k、v的数量默认为8， args.n_heads默认为32
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        # self.n_local_heads = args.n_heads // model_parallel_size
        # self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 每几个q对应一个k、v
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每一个head 的权重矩阵为[dim, head_dim],dim=4096,heads=32
        self.head_dim = args.dim // args.n_heads
        self.norm = nn.LayerNorm(args.dim, dtype=dtype,eps=args.norm_eps)
        # self.norm = RMSNorm(args.dim,dtype=dtype ,eps=args.norm_eps)
        # wq [input:4096, output:4096]
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=dtype)
        # wk&wv [input:1024, output:4096]
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=dtype)
        if not is_train:
            print("init kv cache!")
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                ), dtype=dtype
            ).cuda()
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                ), dtype=dtype
            ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):

        output = self.norm(x)
        # output.shape=[bsz, seqlen, embedding_dim]
        bsz, seqlen, _ = output.shape

        xq, xk, xv = self.wq(output), self.wk(output), self.wv(output)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if not self.is_train:
            # Tensor.to(tensor), 可以改变张量的dtype和device，使其与传入的tensor的一致
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)


        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)


        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output + x



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, norm_eps: float, dtype=torch.bfloat16):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=norm_eps, dtype=dtype)
        self.up_fc = Conv1D(hidden_dim, dim, dtype=dtype)
        self.down_fc = Conv1D(dim, hidden_dim, dtype=dtype)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        output = self.norm(x)
        # output = self.norm(output)
        output = self.up_fc(output)
        output = F.silu(output)
        output = self.down_fc(output)
        output = self.dropout(output)
        return output+x



class MY_Embedding(nn.Module):
    def __init__(self, args: ModelArgs, is_train,dtype=torch.bfloat16):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.dim, dtype=dtype)
        self.drop = nn.Dropout(0.1)

    def forward(self, tokens,):
        out = self.embed(tokens)
        out = self.drop(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, is_train,dtype=torch.bfloat16):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, is_train,dtype)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            norm_eps=args.norm_eps,
            dtype=dtype
        )
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        LBL_function=None,
        block_idx=None
    ):
        if LBL_function is not None:
            out = LBL_function(self.attention, x, start_pos, freqs_cis, mask)
            out = LBL_function(self.feed_forward, out)
            return out

        out = self.attention(x, start_pos, freqs_cis, mask)
        out = self.feed_forward(out)
        return out



class FinalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        norm_eps: float,
        bias: bool = False,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        # dtype = dtype or torch.bfloat16
        self.norm = nn.LayerNorm(dim, eps=norm_eps, dtype=dtype)
        # self.norm = RMSNorm(dim,dtype=dtype, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=bias, dtype=dtype)

    def forward(self, x):
        return self.output(self.norm(x))




