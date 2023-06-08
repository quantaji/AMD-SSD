~/.conda/envs/forl-proj/bin/python scripts/amdppo_matrix_game.py \
    --game_type "PrisonerDilemma" \
    --cp_r_max 3.0 \
    --exp_name "with_amd_mlp_model_softmax_assump" \
    --model "mlp" \
    --param_assump "softmax"
