#!/bin/bash

deepspeed --num_gpus=2 shurale/cli/train.py --use_gradient_checkpointing True --deepspeed_stage 2 --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 --use_flash_attention_2 False \
  --load_in_4bit True --apply_lora True --raw_lora_target_modules all --per_device_train_batch_size 8 \
  --warmup_steps 25 --max_steps 150 --save_total_limit 0 --push_to_hub True --hub_model_id BobaZooba/Shurale7B-v1-LoRA-Test \
  --hub_private_repo True --report_to_wandb True --logging_steps 1 --save_steps 25 --path_to_env_file ./.env