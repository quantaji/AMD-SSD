python scripts/amdppo_wolfpack.py \
    --cp_r_max 0.05 \
    --exp_name "amd-cum-r-immediate_model-lstm_assump-softmax" \
    --model "lstm" \
    --param_assump "softmax" \
    --cum_reward true \
    --amd_schedule_half -10000000 \
    --seed 1234
#  90s per iteration
