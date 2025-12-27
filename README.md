The bug is related to the code from line 225 to 333 in the 'async_fsdp.py'.

If we change `async_op=True` to `async_op=False` on line 283, the program runs more or less normally.
Additionally, I also used `async_op=True` in the `all_gather_param` function, but it doesn't affect the program's normal execution. This means the bug only occurs when using asynchronous transfer of DTensors in the backward hook.
Furthermore, this bug causes some tensors to fail to be released, leading to a continuous increase in GPU memory usage.
Finally, I found that DTensor's asynchronous communication seems to have numerical instability issues, which require further investigation.
