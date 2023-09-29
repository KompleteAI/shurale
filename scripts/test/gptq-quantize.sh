#!/bin/bash

python3 shurale/cli/gptq_quantize.py --model_name_or_path ./fused_model/ --apply_lora False --stabilize False \
  --quantization_max_samples 100000 --quantized_model_path ./quantized_model/ --prepare_dataset False \
  --quantized_hub_model_id BobaZooba/Shurale7B-v1-GPTQ-Test --quantized_hub_private_repo True \
  --low_cpu_mem_usage True --path_to_env_file ./.env
  