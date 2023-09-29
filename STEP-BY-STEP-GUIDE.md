# How to train 🌿 Shurale7B-v1: Narrative based chit-chat model

Developed by [@BobaZooba](https://huggingface.co/BobaZooba)

I'm open to work &
partnership: [My CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [My LinkedIn](https://www.linkedin.com/in/boriszubarev/) | [Advising](https://komplete.framer.ai)

[GitHub Repo](https://github.com/KompleteAI/shurale) | Model based
on [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

# Project overview

All the code is located here: `shurale/cli/*`. Everything needed, including the dataset, has been implemented in
the `xllm` project, so to train Shurale7B-v1, all you need to do is run the scripts from the command line.
So, the most important thing is found in the `scripts` folder. These are bash scripts for all necessary actions in the
project.

| Script name          | Purpose                                                                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `download.sh`        | Download and prepare data, download model.                                                                                              |
| `train.sh`           | Train the model (this is usually done using LoRA).                                                                                      |
| `train-deepspeed.sh` | Train the model (this is usually done using LoRA) using multi-gpu and DeepSpeed. Specify the correct number of GPUs used in the script. |
| `fuse.sh`            | Fuse LoRA and upload fused model to Huggingface Hub                                                                                     |
| `gptq-quantize.sh`   | GPTQ quantization of the fused model. Optional step.                                                                                    |

# Steps to reproduce

## 0. Run the correct environment

Cuda version must be >= 11.8

Good docker image: `huggingface/transformers-pytorch-gpu:latest`

## 1. Install X—LLM and other requirements

Install requirements: latest `transformers` and `xllm` (including `deepspeed`, `flash-attn` and `auto-gptq`)

```bash
pip install -r requirements-train.txt
```

Or

```bash
make install-train
```

## 2. Prepare .env file

You can find a file `.env.template` with the content:

```bash
HUGGING_FACE_HUB_TOKEN=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=
TOKENIZERS_PARALLELISM=false
```

Make a `.env` file with filled variables. `HUGGING_FACE_HUB_TOKEN` is necessary, everything about wandb is only needed
for training. These variables will be loaded into the environment when any of the scripts are called.

## 3. Корректные переменные

## 3. Give access permissions to the `scripts/*`

To run bash scripts you need to give them permissions.

```bash
chmod 755 -R ./scripts/
```

Or

```bash
make scripts-access
```

## 3. Prepare dataset and model

Before you begin the training process, it's essential that you first download the necessary data and the model. The
training won't proceed without this as it's during this stage that your training data gets prepared. We've separated
this as a distinct step for a few reasons, chief among them being: if you're utilizing distributed training across
several GPUs - for instance via DeepSpeed - you would otherwise end up redundantly downloading the dataset and model on
each individual GPU, even though you only need to do it once.

Run this command line:

```bash
make download
```

Or

```bash
./scripts/download.sh
```

Or directly

```bash
python3 shurale/cli/download.py \
  --dataset_key soda \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --path_to_env_file ./.env
```

## 4. Train model

For more information about configuring your setup, refer to the [X—LLM](https://github.com/KompleteAI/xllm)
documentation. Make sure to fill in the correct values, such as your specific `hub_model_id` and other relevant
settings.

### Single GPU

Run this command line:

```bash
make train
```

Or

```bash
./scripts/train.sh
```

Or directly

```bash
python3 shurale/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 0 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 2 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --path_to_env_file ./.env
```

### Multiple GPU

This project utilizes DeepSpeed's Stage 2 offloading to facilitate multi-GPU training. However, you have the flexibility
to modify this setting via the config arguments. For instance, you can switch the setting to use `fsdp` if needed.

Also, please specify the correct number of GPUs used in the script.

Run this command line:

```bash
make deepspeed-train
```

Or

```bash
./scripts/train-deepspeed.sh
```

Or directly

```bash
deepspeed --num_gpus=8 shurale/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 2 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 8 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --path_to_env_file ./.env
```

## 5. Fuse LoRA

Если вы обучали модель с использованием LoRA, то вам следует

If you trained the model using LoRA

Run this command line:

```bash
make fuse
```

Or

```bash
./scripts/fuse.sh
```

Or directly

```bash
python3 shurale/cli/fuse.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_model_id BobaZooba/Shurale7B-v1 \
  --hub_private_repo True \
  --force_fp16 True \
  --fused_model_local_path ./fused_model/ \
  --path_to_env_file ./.env
```

## 6. [Optional] GPTQ quantization

Run this command line:

```bash
make gptq-quantize
```

Or

```bash
./scripts/gptq-quantize.sh
```

Or directly

```bash
python3 shurale/cli/gptq_quantize.py \
  --model_name_or_path ./fused_model/ \
  --apply_lora False \
  --stabilize False \
  --quantization_max_samples 100000 \
  --quantized_model_path ./quantized_model/ \
  --prepare_model_for_kbit_training False \
  --quantized_hub_model_id BobaZooba/Shurale7B-v1-GPTQ \
  --quantized_hub_private_repo True \
  --low_cpu_mem_usage \
  --path_to_env_file ./.env
```

## 7. 🎉 Done! You are awesome!

Now your model was trained, fused (and maybe quantized) and saved to HuggingFace Hub.

You can load the model using `transformers`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("BobaZooba/Shurale7B-v1")
model = AutoModelForCausalLM.from_pretrained("BobaZooba/Shurale7B-v1")
```

## 💼 If you want models as cool as this one

### X—LLM

The training of this model utilized the [X—LLM](https://github.com/KompleteAI/xllm) library. This tool makes it easy to
finetune large language models using cutting-edge methods like bitsandbytes int4, QLoRA, DeepSpeed, Flash Attention 2,
and so on. You can effortlessly integrate this library into your projects.

### Advisor

And if your team is hunting for the insights of an adept advisor to propel your projects forward, don't hesitate to
reach out through this website: https://komplete.framer.ai

### New team member

Are you seeking a dynamic addition to your team who possesses the prowess and the know-how to train such innovative
models? Then consider
sharing [my CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing)
or [LinkedIn](https://www.linkedin.com/in/boriszubarev/) with your manager.
