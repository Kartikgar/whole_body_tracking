#!/bin/bash
# BlinkMimic released recipe (blinkmimic/scripts/presets.sh PRESET[main]) for the Isaac port, minus the MuJoCo-Warp
# systems flags (--collect_graph --reset_masked --eprobe_graph --eprobe_async) and the evaluation-probe allocation
# (--eprobe_share/--eprobe_start/--eprobe_every/--eprobe_decay: needs a second simulator, not ported yet).
declare -A PRESET
PRESET[main]="--pi_ema 0.995 --no_sync --fused_adam --fused_rot --compile_more --lr 4e-4 --wm_batches 5 --wm_hidden 1024,1024,1024 \
--fail_reset 0.3 --contact_reward 0.25 --wm_structured --wm_mirror 0.5 --pi_opt muon --wm_opt muon --pi_ref_skip --wm_only_iters 0 \
--bin_fix --ref_contact_fix --ckpt_minutes 0 --cuda_graph --graph_early --pi_batch 2048 --pi_lr_mult 4 --lr_decay_ret 0.75 \
--lr_decay_len 300 --lr_decay_by 1400 --ckpt_every 100 --ckpt_light --no_policy_graph"
# Full recipe on Isaac: + the evaluation-probe allocation of PRESET[main] (probe worlds share the training scene;
# eager sweeps; the training worlds are snapshotted / restored around each sweep).
PRESET[eprobe]="${PRESET[main]} --eprobe_share 0.15 --eprobe_start 1000 --eprobe_every 125 --eprobe_decay 0.7"
