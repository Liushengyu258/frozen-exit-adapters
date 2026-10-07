# Experimental protocol

- Frozen teacher: Qwen3.8-27B text path, 64 blocks, width 5120; frozen final RMSNorm and output head. No model revision available in archive.
- Sources: retained depths 60, 61, 62, 63; target: post-block-64 residual before final normalization.
- Corpus: 2250 GSM8K math prompts, 2250 Code Alpaca Python prompts, 500 procedural controls; split 4000/500/500 by prompt family. Max 32 aligned positions per prompt (up to 8 prompt and 24 answer positions).
- Adapter: A(x)=x+s f(x/s); s fixed from first saved positions of the first 512 training prompts; biases included; output projection initialized to zero.
- Search: widths 6144, 8192, 10240, 12288 × SiLU-1, GELU-1, SiLU-2, SwiGLU. All hidden widths exceed 5120.
- Loss: mean squared error after the frozen teacher RMSNorm; sampling uniformly over prompts then saved positions. No labels, tests, logits or reward in adapter loss.
- Optimization: AdamW, batch 512, no weight decay, gradient norm clip 1, FP32 parameters with BF16 autocast, seed 20261006 + source depth. Timed warmup/cosine learning rate, 3e-4 screening and 8e-5 refinement. Exact active-time quotas in training_plan.json. No patience early stopping.
- Selection: lowest vector validation MSE; top two per depth refined, same validation criterion across phases. Selected candidate plus explicitly requested SiLU-1/10240 evaluated at each depth, plus identity-61/62.
- Refinement: screen checkpoints stored in BF16, optimizer restarted. This rounding is part of the recorded procedure.
- Generation: greedy, thinking disabled, batch 4, 256 new-token cap; 100 GSM8K and 99 supported MBPP questions. Shared output head; actual prefix execution checked. Unexecuted suffix weights remain resident.
- Statistics: paired-question percentile bootstrap, 20000 resamples, seed 20261007; exploratory, unadjusted, no training-seed uncertainty. Same held-out generation test had been inspected in the earlier exploratory round.
- Four depth winners were SiLU-2 width 12288. Longer fitting reduced training loss but increased late validation loss. No global convergence, intrinsic-nonlinearity, cross-model stitching or deployment-speed claim.
