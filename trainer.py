import json
import time
from typing import List, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F
from blocks import ModelArgs
from GPT_model import Transformer
import torch.cuda.amp
from transformers import GPT2Tokenizer



def build_model(
        config_path: str = "./params.json",
        max_seq_len: int = 2048,
        max_batch_size: int = 4,
        is_train=False,
        model_dtype=torch.bfloat16,
        dtype_bit16=torch.bfloat16,
        dtype_bit32=torch.float32,
):
    with open(config_path, "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    # print(params)
    start_time = time.time()
    # tokenizer = Tokenizer(model_path=tokenizer_path)
    # assert model_args.vocab_size == tokenizer.vocab_size
    print(f"Initializing ...")
    model = Transformer(model_args, is_train=is_train,model_dtype=model_dtype,
                        dtype_bit16=dtype_bit16,
                        dtype_bit32=dtype_bit32, )
    # print(model)
    print(f"The model is initialized in {time.time() - start_time:.2f} seconds.")
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"The model size is: {num_parameters / (1000 ** 3):.2f} Billion.")

    return model

def build_tokenizer(
        tokenizer_path: str = "gpt2",
):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer




class GPT2_Trainer:
    def __init__(self, model: Transformer, tokenizer: GPT2Tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def text_train(
            self,
            prompts: List[str],
            device=torch.device("cpu"),
            rank: int = -1,
    ):
        batch_size = self.model.params.max_batch_size
        if batch_size <= 8 and batch_size > 0:
            prompts = prompts[:batch_size]
        else:
            raise TypeError("The batch size should be <= 8 and > 0!")

        prompt_tokens = self.tokenizer(prompts, truncation=True, padding='max_length',
                                       max_length=self.model.params.max_seq_len)
        prompt_tokens = prompt_tokens['input_ids']
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_prompt_len + 1
        pad_id = 50256
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device='cuda')
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        _target_tokens = tokens[:, 1:max_prompt_len + 1].clone().detach().contiguous()
        _input_tokens = tokens[:, prev_pos:max_prompt_len].clone().detach().contiguous()
        print(f"Input tokens shape: {_input_tokens.shape}.")
        print(f"Target tokens shape: {_target_tokens.shape}.")

        # prof = torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=6, warmup=6, active=3, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./pt_log/bsz={bsz}_ombb_v3_single_offload_v3_dim2048'),
        #     record_shapes=True,
        #     with_stack=True)
        # prof.start()

        start = time.time()
        mystep = -1
        max_steps = 10
        for i in range(30):
            # prof.step()
            inputs_tokens = _input_tokens.clone()
            target_tokens = _input_tokens.clone()
            # with torch.amp.autocast(device_type='cuda'):
            logits = self.model(inputs_tokens, prev_pos)
            temp_shape = logits.size()
            loss = F.cross_entropy(
                input=logits.view(-1, temp_shape[2]),
                target=target_tokens.view(-1),
                reduction="mean",
                ignore_index=50256,
                label_smoothing=0.01)
            loss.backward()

            mystep += 1
            if mystep == 0:
                print(f"Total training steps: {max_steps}.")
                print(f"====================Training Step and Loss====================")
                torch.cuda.synchronize()
                start = time.time()
            if mystep == max_steps:
                print(f"====================Training End==============================")
                torch.cuda.synchronize()
                Total_time = time.time() - start
                print(f"Total training time: {Total_time:.2f} seconds.      Total training steps: {mystep}.")
                print(f"Average time cost for a training step: {Total_time / mystep:.2f} seconds.")
                print(
                    f"Throughput: {mystep * bsz * self.model.params.max_seq_len / Total_time:.2f} tokens per seconds.")
                # prof.stop()
                break

            if rank==0:
                print(f"Step = {mystep}, Loss: {loss}.")
            #
        # print(torch.cuda.memory_summary())
        print(f"Max memory usage: {torch.cuda.max_memory_reserved() / (1024 ** 3)} GiB.")
        return 10086
