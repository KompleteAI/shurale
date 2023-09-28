#!/bin/bash

python3 shurale/cli/gptq_quantize.py --model_name_or_path ./fused_model/ --apply_lora False --stabilize False \
  --quantization_max_samples 100000 --quantized_model_path ./quantized_model/ \
  --quantized_hub_model_id KompleteAI/Shurale7b-v1-GPTQ --quantized_hub_private_repo True --path_to_env_file ./.env
  