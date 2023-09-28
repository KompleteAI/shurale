#!/bin/bash

python3 shurale/cli/fuse.py --model_name_or_path mistralai/Mistral-7B-v0.1 --lora_hub_model_id KompleteAI/Shurale7B-V1-LoRA \
  --hub_model_id KompleteAI/Shurale7B-V1 --hub_private_repo True --force_fp16 True --fused_model_local_path ./fused_model/ \
  --path_to_env_file ./.env
