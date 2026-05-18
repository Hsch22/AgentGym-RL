# TextCraft 2xH200 Baseline Collapse Analysis

## Summary

This note documents why the `textcraft_scalinginter_baseline_2xh200_fastsettings_20260516_041512`
run collapsed even though the original 8-GPU TextCraft ScalingInter baseline had been reproduced.

The short conclusion is:

> Matching total batch size is not sufficient to make PPO/FSDP/vLLM training numerically equivalent across
> different GPU counts. In this run, the 2xH200 configuration changed the per-GPU trajectory load,
> token batching, FSDP reduction behavior, and vLLM rollout scheduling. The run remained healthy through
> the 10-round and most of the 20-round phase, then became unstable after entering the 30-round phase and
> eventually collapsed into invalid repeated-token outputs.

This is not evidence that TextCraft cannot be reproduced. It is evidence that the current 2-GPU
`fastsettings` run is not a controlled reproduction of the original 8-GPU baseline.

## Relevant Runs

### Stable 8-GPU Baseline

Local W&B history:

```text
results/wandb_agentgym_rl/runs/01_c8u3q5ae_textcraft_scalinginter_baseline_3b_20260420_1459/history.csv
```

Eval history:

```text
results/wandb_agentgym_rl_eval/runs/01_9oeixbod_textcraft_eval_textcraft_scalinginter_baseline_3b_20260420_1459_20260421_011357/history.csv
```

Key outcome:

- Step 329 train score: `0.969`
- Step 329 entropy loss: `0.397`
- Step 329 KL loss: `0.222`
- Step 329 response length mean: `311`
- Eval final `avg_at_1/pass_at_1`: `0.87/0.87`
- No NaN collapse in the available W&B history.

### Collapsed 2xH200 Baseline

Training log:

```text
logs/textcraft_scalinginter_baseline_2xh200_fastsettings_20260516_041512.log
```

W&B run:

```text
https://wandb.ai/hsch224-peking-university/agentgym-rl-textcraft/runs/vysvz717
```

Local offline run:

```text
AgentGym-RL/wandb/offline-run-20260516_041717-vysvz717
```

Key outcome:

- Step 200 still healthy: score `0.941`, candidate string-valid ratio `0.994`.
- Step 225 begins to drift: score `0.910`, candidate string-valid ratio `0.864`, grad norm `7.711`.
- Step 234 shows numerical explosion: `pg_loss=142787.414`, `grad_norm=7.94e8`.
- Step 250 is already severely degraded: score `0.457`, candidate string-valid ratio `0.131`, KL loss `1.863`.
- Step 274 is the first NaN point in actor metrics.
- Step 275 onward collapses: score `0`, candidate string-valid ratio `0`, response length mean `1950`, actor losses/grad are NaN.

## Configuration Difference

The original 8-GPU baseline and the 2xH200 run both used:

- `train_batch_size=32`
- `rollout.n=8`
- total trajectories per step: `32 * 8 = 256`
- `rounds=[10,20,30]`
- `steps_scaling_inter=100`
- `max_prompt_length=512`
- `max_response_length=10240`
- `rollout.max_tokens=512`
- `ppo_epochs=2`
- `ppo_mini_batch_size=8`
- `lr=1e-6`
- `kl_coef=0.001`

But the GPU count changed:

| Run | GPUs | Total trajectories/step | Approx trajectories/GPU |
| --- | ---: | ---: | ---: |
| 8-GPU baseline | 8 | 256 | 32 |
| 2xH200 fastsettings | 2 | 256 | 128 |

The 2xH200 run also used more aggressive per-GPU token settings:

| Setting | 8-GPU baseline | 2xH200 fastsettings |
| --- | ---: | ---: |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | `32768` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.7` | `0.8` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | not explicitly set in the old script | `16384` |
| `trainer.n_gpus_per_node` | `8` | `2` |

Therefore, the 2-GPU run did not preserve the original per-rank trajectory or token load. It preserved total
trajectories, while each GPU handled roughly 4x as many rollout trajectories.

## Observed Failure Timeline

The collapse aligns with the transition into the `30`-round regime.

| Step range | Max rounds | Status |
| --- | ---: | --- |
| `1-99` | 10 | Learning normally, score rises from near zero to around `0.8`. |
| `100-199` | 20 | Mostly healthy, score around `0.89` average; step 175 reaches `0.969`. |
| `200-224` | 30 | Initially still healthy, but KL and valid-ratio drift begins. |
| `225-249` | 30 | Instability becomes visible: valid ratio drops, grad norm spikes, KL rises. |
| `250-274` | 30 | Severe degradation: invalid outputs dominate, PPO metrics show large spikes. |
| `275+` | 30 | Full collapse: all sampled responses become repeated `!` tokens. |

Representative training metrics:

| Step | Score mean | KL loss | Grad norm | Candidate string-valid ratio | Response length mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | `0.941` | `0.124` | `0.314` | `0.994` | `333` |
| 225 | `0.910` | `0.375` | `7.711` | `0.864` | `338` |
| 234 | `0.824` | `0.205` | `7.94e8` | `0.651` | `268` |
| 238 | `0.508` | `1.176` | `6.20e9` | `0.287` | `228` |
| 250 | `0.457` | `1.863` | `527.726` | `0.131` | `140` |
| 274 | `0.770` | `nan` | `nan` | `0.839` | `710` |
| 275 | `0.000` | `nan` | `nan` | `0.000` | `1950` |
| 330 | `0.000` | `nan` | `nan` | `0.000` | `1950` |

## Rollout Evidence

The executor logs show that the model did not merely become less accurate. It collapsed into malformed
token repetition.

Before collapse, step 225 still emits normal ReAct-style responses:

```text
Thought: To craft a smithing table, I need 4 planks and 2 iron ingots...

Action: get 4 planks
```

By step 250, malformed prefixes and invalid responses appear:

```text
ouser

Thought: I have obtained a lily of the valley...

Action: craft 1 white dye using 1 lily of the valley
```

By step 275, all sampled responses become repeated exclamation marks:

```text
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

At step 275:

- `7680/7680` responses have no `Action:`.
- Mean raw response length is exactly `512`, the per-turn generation cap.
- Candidate string-valid ratio is `0`.

This is consistent with actor distribution collapse, not with an environment-server failure.

## Why Total Batch Matching Is Not Enough

In ideal math, if the rollout samples, old log probabilities, advantage values, minibatch order,
optimizer state, floating-point operations, and random seeds were exactly identical, then changing the
number of GPUs would not change the PPO update.

The actual training stack does not satisfy these assumptions.

### 1. PPO Is Nonlinear and Sensitive to Rollout Differences

PPO optimizes a clipped objective:

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))

L = min(r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t)
```

Small changes in generated rollouts, old log probabilities, or advantage normalization can change which
tokens are clipped. Once different samples cross the clipping boundary, the update is not a simple linear
average of per-sample gradients.

### 2. GRPO Advantage Is Group-Based

The run uses:

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n=8
```

GRPO computes outcome advantages within repeated rollout groups for the same prompt. If vLLM generation
or batching changes which samples succeed or fail in a prompt group, the whole group's standardized
advantage changes.

This makes on-policy trajectory differences especially important.

### 3. FSDP and Dynamic Token Batching Change Floating-Point Reduction Order

The 2-GPU run increased per-rank token load and used dynamic batch sizing:

```text
actor_rollout_ref.actor.use_dynamic_bsz=True
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
```

Different GPU counts change:

- per-rank batch composition,
- padding/removal layout,
- dynamic microbatch boundaries,
- gradient accumulation order,
- FSDP reduce-scatter/all-reduce order.

Floating-point addition is not associative:

```text
(a + b) + c != a + (b + c)
```

The effect is larger with bf16 and long sequence batches.

### 4. vLLM Rollout Scheduling Is Not Guaranteed to Be GPU-Count Invariant

vLLM generation depends on batching, scheduling, KV-cache state, chunked prefill, and worker placement.
Changing from 8 GPUs to 2 GPUs changes the request scheduling even when the logical total number of
rollouts is the same.

Because PPO is on-policy, small rollout differences change the next optimization step, and the difference
can compound over hundreds of updates.

### 5. The 30-Round Phase Amplifies Errors

Moving from 20 to 30 rounds increases:

- sequence length,
- prompt history length,
- number of sampled actions per prompt,
- probability of malformed response contamination,
- KL pressure from longer generations.

The observed failure begins after the 30-round phase starts, which matches this risk profile.

## External References

The exact TextCraft collapse is local to this experiment, but the underlying mechanisms are discussed in
community issues and official documentation.

### verl Batch Semantics

verl issue discussion notes that PPO/GRPO batch parameters in FSDP settings have important per-GPU or
per-rank semantics. This supports the point that preserving total batch size alone is not enough.

Reference:

```text
https://github.com/verl-project/verl/issues/2266
```

### vLLM Reproducibility

vLLM documentation states that default behavior is not generally reproducible and that reproducibility is
limited to the same hardware and software conditions.

Reference:

```text
https://docs.vllm.ai/en/v0.10.2/usage/reproducibility.html
```

There is also a vLLM community issue reporting nondeterministic outputs under concurrent requests even
with deterministic-looking settings.

Reference:

```text
https://github.com/vllm-project/vllm/issues/23138
```

### PyTorch Reproducibility

PyTorch documentation states that exact reproducibility is not guaranteed across devices, releases, or
platforms, and that deterministic operation often requires explicit controls with performance tradeoffs.

Reference:

```text
https://docs.pytorch.org/docs/stable/notes/randomness.html
```

### LLM Inference Sensitivity to Batch/GPU Count

Recent work also analyzes how batch size, GPU count, and GPU version can affect LLM inference outputs due
to low-precision floating-point non-associativity and scheduling differences.

Reference:

```text
https://arxiv.org/abs/2506.09501
```

## Most Likely Cause

The most likely cause is not a single bug but a system-level instability:

1. The 2-GPU run preserved total trajectories but increased per-GPU trajectory load by about 4x.
2. The 2-GPU run also increased per-GPU PPO token budget and rollout memory pressure.
3. Entering the 30-round regime made sequences longer and noisier.
4. PPO/GRPO amplified early rollout and numerical differences.
5. KL and gradient metrics spiked, then actor outputs collapsed into repeated punctuation.

The sharp transition from normal ReAct outputs to all-`!` outputs indicates actor distribution collapse.

## What Not To Conclude

Do not conclude:

- TextCraft cannot be reproduced.
- The original 8-GPU baseline was invalid.
- The final `global_step_330` checkpoint from the 2-GPU run is meaningful.
- Matching total trajectories per step is sufficient for a controlled baseline.

The 2-GPU fastsettings run should be treated as an unstable throughput experiment, not as a valid
baseline reproduction.

## Recommended Next Steps

### Use Stable Checkpoints Only

For analysis and eval, avoid the collapsed checkpoints:

```text
275, 300, 325, 330
```

Useful checkpoints around the transition:

```text
175, 200, 225, 250
```

### Reproduce Baseline on 2 GPUs by Matching Per-GPU Load

For a controlled 2-GPU baseline, first match the 8-GPU per-rank load instead of the total trajectory count:

```text
TEXTCRAFT_TRAIN_BATCH_SIZE=8
TEXTCRAFT_ROLLOUT_N=8
total trajectories = 64
trajectories per GPU = 32
TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU=16384
TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
```

This should be the first stability target.

### Scale Throughput Gradually

After a stable 2-GPU baseline is obtained, increase throughput in controlled increments:

1. `train_batch_size=8`, `rollout.n=8`
2. `train_batch_size=16`, `rollout.n=8`
3. `train_batch_size=24`, `rollout.n=8`
4. `train_batch_size=32`, `rollout.n=8`

At each stage, monitor:

- `critic/score/mean`
- `actor/kl_loss`
- `actor/grad_norm`
- `actor/entropy_loss`
- `rollout/candidate_string_valid_ratio`
- `response_length/mean`
- executor samples from `executer_logs`

Stop or reduce scale if any of the following appears:

- candidate string-valid ratio drops sharply,
- KL loss rises above the historical baseline band,
- grad norm spikes by orders of magnitude,
- malformed prefixes or repeated punctuation appear,
- response length hits the per-turn cap for most samples.

### Keep Baseline and G2RL Under the Same Stable System Configuration

For algorithm comparison, use the same stable system configuration for both baseline and G2RL. Otherwise,
it is impossible to separate algorithmic effects from distributed-system instability.
