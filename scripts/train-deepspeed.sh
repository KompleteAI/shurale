#!/bin/bash

deepspeed --num_gpus=8 shurale/cli/train.py --use_gradient_checkpointing True --deepspeed_stage 2 --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 --use_flash_attention_2 Flase \
  --load_in_4bit True --apply_lora True --raw_lora_target_modules all --per_device_train_batch_size 6 \
  --warmup_steps 1000 --save_total_limit 0 --push_to_hub True --hub_model_id KompleteAI/Shurale7b-v1-LoRA \
  --hub_private_repo True --report_to_wandb True --path_to_env_file ./.env