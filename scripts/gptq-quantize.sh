#!/bin/bash

python shurale/cli/gptq_quantize.py --model_name_or_path KompleteAI/ShuraleTest --apply_lora False --stabilize False \
  --quantization_max_samples 100000 --quantized_model_path ./quantized_model/ \
  --quantized_hub_model_id KompleteAI/ShuraleTestGPTQ --quantized_hub_private_repo True
  