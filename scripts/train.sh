#!/bin/bash

python shurale/cli/train.py --use_gradient_checkpointing True --deepspeed_stage 2 --stabilize True \
  --model_name_or_path BobaZooba/Shurale7b-v1 --model_type llama --use_flash_attention_2 True \
  --load_in_4bit True --apply_lora True --raw_lora_target_modules all --per_device_train_batch_size 2 \
  --warmup_steps 100 --max_steps 500 --save_total_limit 0 --push_to_hub True --hub_model_id KompleteAI/ShuraleTestLoRA \
  --hub_private_repo True --report_to_wandb True
