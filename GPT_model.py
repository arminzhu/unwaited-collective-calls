import torch
from torch import nn
from blocks import ModelArgs, TransformerBlock, precompute_freqs_cis, FinalBlock
from async_fsdp import Async_FSDP, L_F

class Transformer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 is_train: bool = False,
                 model_dtype=torch.bfloat16,
                 dtype_bit16=torch.bfloat16,
                 dtype_bit32=torch.float32,
                 ):
        super().__init__()
        self.params = params
        self.is_train = is_train
        self.model_dtype = model_dtype
        self.dtype_bit16 = dtype_bit16
        self.dtype_bit32 = dtype_bit32

        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.is_train = is_train
        self.split_block = False
        self.block_nums = params.n_layers + 2

        self.embed_tokens = nn.Embedding(num_embeddings=params.vocab_size, embedding_dim=params.dim,
                                         dtype=model_dtype)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, is_train, dtype=model_dtype))

        self.output = FinalBlock(params.dim, params.vocab_size, params.norm_eps, bias=False, dtype=model_dtype)
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def ombb_initialize(self, model_info, gpu_mesh=None, cpu_mesh=None, rank=0, device="cpu",
                        train_lr=0.0005, alpha=0.9, beta=0.95, eps=1e-8, weight_decay=1e-2, ):
        L_F.initialize(model_info, gpu_mesh=gpu_mesh, cpu_mesh=cpu_mesh, rank=rank, device=device,
                       lr=train_lr, alpha=alpha, beta=beta, eps=eps, weight_decay=weight_decay,)

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        tokens = tokens.to(dtype=torch.float32, device=tokens.device)
        tokens.requires_grad = True
        h = Async_FSDP(self.embed_tokens, tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        for layer in self.layers:
            if self.split_block:
                h = layer(h, start_pos, freqs_cis, mask, LBL_function=Async_FSDP)
            else:
                h = Async_FSDP(layer, h, start_pos, freqs_cis, mask)
        h = Async_FSDP(self.output, h)
        return h


if __name__ == '__main__':
    multiple_of = 1024.
    hidden_dim = 4096. * 4.
    ffn_dim_multiplier = 1.3
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    print(multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of))
    # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
