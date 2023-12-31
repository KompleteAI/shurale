# How to train 🌿 Shurale7B-v1: Narrative based chit-chat model

Developed by [@BobaZooba](https://www.linkedin.com/in/boriszubarev/) |
E-mail: [bobazooba@gmail.com](mailto:bobazooba@gmail.com)

Open for
partnership: [Advising](https://komplete.framer.ai) | [CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [LinkedIn](https://www.linkedin.com/in/boriszubarev/)

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/JudU3rrPP5i87CfwINANO.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/KompleteAI/xllm)

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

Please check these useful materials:

- [Detailed guide](https://github.com/KompleteAI/xllm/blob/main/DETAILED_GUIDE.md): here, we go into detail about
  everything the library can do
- [Demo project](https://github.com/KompleteAI/xllm-demo): here's a step-by-step example of how to use X—LLM and fit it
  into your own project
  - [Minimal demo project](https://github.com/KompleteAI/xllm-demo-minimal): simplest project using X—LLM
- [Template project](https://github.com/KompleteAI/xllm-template): here's a template, a kickoff point you can use for
  your projects

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

## 3. Set up correct config

You need to set the correct config variables in all the scripts used (download, train, etc.). For example, the correct
name of your project in HuggigFace Hub and so on.

### Useful materials

- [Important config fields for different steps](!link)
- [How do I choose the methods for training?](!link)
- [Detailed description of all config fields](!link)

## 4. Give access permissions to the `scripts/*`

To run bash scripts you need to give them permissions.

```bash
chmod 755 -R ./scripts/
```

Or

```bash
make scripts-access
```

## 5. Prepare dataset and model

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

## 6. Train model

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
  --gradient_accumulation_steps 4 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --save_safetensors True \
  --use_gradient_checkpointing True \
  --stabilize True \
  --max_length 2048 \
  --use_flash_attention_2 False \
  --prepare_model_for_kbit_training True \
  --load_in_4bit True \
  --apply_lora True \
  --label_smoothing_factor 0.1 \
  --path_to_env_file ./.env
```

#### Pay attention to fields

- hub_model_id

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
  --deepspeed_stage 2 --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/Shurale7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --save_safetensors True \
  --use_gradient_checkpointing True \
  --stabilize True \
  --max_length 2048 \
  --use_flash_attention_2 False \
  --prepare_model_for_kbit_training True \
  --load_in_4bit True \
  --apply_lora True \
  --label_smoothing_factor 0.1 \
  --path_to_env_file ./.env
```

#### Pay attention to fields

- hub_model_id

## 7. Fuse LoRA

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
  --max_shard_size 1GB \
  --path_to_env_file ./.env
```

#### Pay attention to fields

- hub_model_id
- lora_hub_model_id

## 8. [Optional] GPTQ quantization

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
  --quantization_max_samples 64 \
  --quantized_model_path ./quantized_model/ \
  --prepare_model_for_kbit_training False \
  --quantized_hub_model_id BobaZooba/Shurale7B-v1-GPTQ \
  --quantized_hub_private_repo True \
  --low_cpu_mem_usage True \
  --path_to_env_file ./.env
```

#### Pay attention to fields

- hub_model_id
- quantized_hub_model_id

## 9. 🎉 Done! You are awesome!

Now your model was trained, fused (and maybe quantized) and saved to HuggingFace Hub.

You can load the model using `transformers`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("BobaZooba/Shurale7B-v1")
model = AutoModelForCausalLM.from_pretrained("BobaZooba/Shurale7B-v1")
```

## 🚀 Call to action

**Looking for an expert in modern LLMs?** I've got the experience you need. I'll guide you through every step,
fine-tuning everything from data collection to model training and improvement.

**Why me?** Well, with six years of experience in deep learning R&D projects, I've mastered a range of roles - from
leading a team to rolling up my sleeves as an engineer. I've built and improved products from scratch and I'm keen to do
the same for you.

**Worried about your team?** Don't be. With four years as a lecturer at Russia’s best university, I can equip them with
the skills they need to succeed.

**Want to know more?** Check
out [my CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing), [LinkedIn](https://www.linkedin.com/in/boriszubarev/),
and [past projects](https://komplete.framer.ai/cases) for the full scoop.

**Ready to start?** Let's arrange a free intro meeting. I'll outline the resources we'll need to make your project a
success.
[Contact me form](https://komplete.framer.ai/#contact)

If you're an engineer, I'd appreciate it if you could pass
along [my LinkedIn](https://www.linkedin.com/in/boriszubarev/) or [website](https://komplete.framer.ai/) to your
manager.
