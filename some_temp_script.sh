#!/bin/bash

actor_n_params=("125m" "350m" "1.3b" "5b")
critic_n_params=("125m" "350m")

for actor_n_param in "${actor_n_params[@]}"; do
    for critic_n_param in "${critic_n_params[@]}"; do
        echo "Running actor_n_param: $actor_n_param, critic_n_param: $critic_n_param"
        nohup python3 -m apps.main start -e opt-${actor_n_param}+${critic_n_param}-chat-rlhf-benchmark -f 20230922-01 >> /dev/null &
    done
done