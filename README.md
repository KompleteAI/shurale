# 🌿 Shurale7b-v1: Narrative based chit-chat model

Developed by [@BobaZooba](https://huggingface.co/BobaZooba)

**I'm open to
work:
** [My CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [My LinkedIn](https://www.linkedin.com/in/boriszubarev/) | [Advising](https://komplete.framer.ai)

[GitHub Repo](https://github.com/KompleteAI/shurale)

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/JudU3rrPP5i87CfwINANO.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/KompleteAI/xllm)

# 🪄 About

| **HuggingFace Hub** | **7b**                                                 | **7b-gptq**                                                 | **13b**     | **13b-gptq** |
|---------------------|--------------------------------------------------------|-------------------------------------------------------------|-------------|--------------|
| **Shurale-v1**      | [Link](https://huggingface.co/KompleteAI/Shurale7b-v1) | [Link](https://huggingface.co/KompleteAI/Shurale7b-v1-GPTQ) | Coming soon | Coming soon  |
| **Shurale-v2**      | Coming soon                                            | Coming soon                                                 | Coming soon | Coming soon  |

<div align="justify">

<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/EmwEd5khHmzUTatA_tXB0.png" alt="Shurale" width="200" height="200" style="float: right; float: bottom; margin-left: 50px;" />

**Shurale** is a dialogue model perfect for engaging in realistic and wide-reaching discussions.
The model's strength comes from its 7-billion parameter foundation (based on **Llama 2**), which was meticulously *
*trained on over a million dialogues**, with clear roles defined for dialogue participants.
Unique in its approach, it uses **narrative context**, aiding in the creation and maintenance of coherent characters
throughout a conversation.
This results in communications that are seamless, lively, and strikingly natural. A pleasant detail: the entire training
process only **cost $100**.

> Shurale [/ʃʊrɑˈlʲe/] is a forest spirit in Bashkir and Tatar mythology.

</div>

[Do you want models as cool as this one?](https://huggingface.co/BobaZooba/Shurale#💼-if-you-want-models-as-cool-as-this-one)

---

# 🤔 Why not ChatGPT?

ChatGPT, even GPT4, struggles with producing human-like dialogues.
A lot of modifications and tricks are necessary just to receive a basic human-like response to a simple query such as "
how are you?".
Responses from GPT models and other instructional models often lack authenticity and feel awkward.
These instructional models thrive in a wholly different sphere related to rational processes and information retrieval,
offering limited control and training opportunities.

**Bottom line** - trying to chat with GPT like it's a human will likely lead to a slew of problems.
**Shurale**, however, tackles this issue effectively.
It steers clear of these hurdles by permitting more human-like dialogues and providing provisions to create a distinct
character.

---

# 📝 Prompt

The parts of the dialogue (narrative and replies) are separated using a newline symbol: **\n**

The **maximum length** during training is **2048 tokens**.

The [SODA](https://huggingface.co/datasets/allenai/soda) dataset was used for the training process.

## Format

Training examples consisted of both the narrative and the dialogue itself, with the participants' names clearly
indicated.

<table>
<tr>
<td>
Narrative
</td>
<td>
A description of the situation within the dialogue
</td>
</tr>
<tr>
<td>
Characters names
</td>
<td>
A list of names of the characters participating in the dialogue
</td>
</tr>
<tr>
<td>
Phrases
</td>
<td>
Phrases used by the participants in the dialogue
</td>
</tr>
</table>

Narratives were deliberately omitted from 5% of the training examples, allowing the model to maintain a dialogue even
without a narrative. However, using the model without a narrative is generally not recommended.

## Example

### Training sample

The baton was passed to Garry who then became the boss. He ran the show with an iron fist, making sure that everything
was done his way. No one dared to cross him for fear of being on the receiving end of his wrath
Garry: What the hell is going on around here? I thought I told you to get this place in order!
Bob: I'm sorry, boss. We've been having some trouble with the employees lately. They just don't seem to be following
orders like they used to.
Garry: Well, you need to get them in line or I'll find someone who will! This place is a mess and it's all your fault!
Bob: Yes, boss. I'll take care of it right away.

### Real world example

The baton was passed to Garry who then became the boss. He ran the show with an iron fist, making sure that everything
was done his way. No one dared to cross him for fear of being on the receiving end of his wrath
Garry: What the hell is going on around here? I thought I told you to get this place in order!
Bob:

In this example, we explicitly tell the model that it's now Bob's turn to speak. The end of the reply can be designated
either by a newline symbol or by the name of the first character followed by a colon (**Garry:**).

---

# 🔧 How to use

## Transformers

1. Load the model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("KompleteAI/ShuraleBase")
model = AutoModelForCausalLM.from_pretrained("KompleteAI/ShuraleBase")
```

2. Run generation

```python
input_text = "Dialog between two colleagues: Emma and Anna.\nEmma:"

tokenized = tokenizer(
  input_text,
  return_tensors="pt"
).to("cuda:0")

generated_indices = model.generate(
  **tokenized,
  do_sample=True,
  max_new_tokens=128,
  repetition_penalty=1.1,  # llama2 is prone to repetitions
  top_p=0.9
)[0].cpu()

print(tokenizer.decode(generated_indices))
```

## Text Generation Inference

Run model as a service using HuggingFace 🤗 inference server:
https://github.com/huggingface/text-generation-inference#get-started

<details>
<summary>1. Start a docker container with the model</summary>

### Docker

```bash
model=KompleteAI/ShuraleBase
volume=$PWD/data
version=1.0.3  # please make sure you are using latest or stable version

docker run --gpus all --shm-size 1g -p 8080:80 -v \
  $volume:/data ghcr.io/huggingface/text-generation-inference:$version \
  --model-id $model --max-batch-prefill-tokens 2048 --dtype bfloat16
```

### RunPod

If you want to run a model at RunPod you can find ready to use template by name "ShuraleBase" at RunPod. Please note
that **port 8081** is used to run this template.

https://www.runpod.io/console/gpu-cloud

| Field             | Value                                                                                                                      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------|
| Container Image   | ghcr.io/huggingface/text-generation-inference:1.0.3                                                                        |
| Docker Command    | --model-id KompleteAI/ShuraleBase --num-shard 1 --port 8081 --max-batch-prefill-tokens 2048 --dtype bfloat16 --json-output |
| Container Disk    | 5                                                                                                                          |
| Volume Disk       | 15                                                                                                                         |
| Volume Mount Path | /data                                                                                                                      |
| Expose HTTP Ports | 8081,8080                                                                                                                  |
| Expose TCP Ports  | 8082                                                                                                                       |

</details>

<details>
<summary>2. Send request to the server and parse the response</summary>

```python
import requests
import json

url = "127.0.0.1:8080/generate"
headers = {"Content-Type": "application/json"}
data = {
  "inputs": "Dialog between two colleagues: Emma and Anna.\nEmma:",
  "parameters": {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "stop": ["\n"]
  }
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json()["generated_text"].strip())
# Hello, Anna! How was your evening?
```

Or

```bash
pip install text-generation
```

```python
from text_generation import Client

input_text = "Dialog between two colleagues: Emma and Anna.\nEmma:"

client = Client("http://127.0.0.1:8080")
print(client.generate(input_text, max_new_tokens=20).generated_text)

text = ""
for response in client.generate_stream(input_text, max_new_tokens=20):
  if not response.token.special:
    text += response.token.text
print(text)
```

</details>

---

# 💼 If you want models as cool as this one

## X—LLM

The training of this model utilized the [X—LLM](https://github.com/KompleteAI/xllm) library. This tool makes it easy to
finetune large language models using cutting-edge methods like bitsandbytes int4, QLoRA, DeepSpeed, Flash Attention 2,
and so on. You can effortlessly integrate this library into your projects.

## New team member

Are you seeking a dynamic addition to your team who possesses the prowess and the know-how to train such innovative
models? Then consider
sharing [my CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing)
or [LinkedIn](https://www.linkedin.com/in/boriszubarev/) with your manager.

## Advisor

And if your team is hunting for the insights of an adept advisor to propel your projects forward, don't hesitate to
reach out through this website: https://komplete.framer.ai

---

# 🚄 Training Process

[<img src="https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/JudU3rrPP5i87CfwINANO.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/KompleteAI/xllm)

## Dataset

The model was trained using only the training part of the [SODA](https://huggingface.co/datasets/allenai/soda) dataset.

## Results

This model, based on llama2-7b, was trained on over 1.1 million dialogues using 4 RTX 4090 (24 Gb) GPUs. The training
process lasted 45 hours and made use of advanced techniques such as QLoRA (int4), DeepSpeed Stage 2, Flash Attention 2,
and gradient checkpointing.

### Overall

| Field                | Value             |
|----------------------|-------------------|
| Model                | llama2-7b         |
| Training steps       | 18500             |
| Warm up steps        | 500               |
| Num training samples | 1,119,582 dialogs |
| Num training tokens  | 300,036,117       |
| Global batch size    | 64                |
| Max batch tokens     | 131,072           |
| Loss                 | 1.96              |
| Perplexity           | 7.1               |
| GPU                  | 4 x RTX 4090      |
| Cost                 | $100              |
| Training time        | 45 hours          |
| Provider             | vast.ai           |

### Important training details

| Field                      | Value         |
|----------------------------|---------------|
| Use gradient checkpointing | True          |
| Use bnb int4               | True          |
| Apply LoRA                 | True          |
| LoRA rank                  | 64            |
| LoRA alpha                 | 32            |
| LoRA layers                | all           |
| Scheduler                  | WarmupDecayLR |
| Max lr                     | 2e-4          |
| Use Flash Attention 2      | True          |
| DeepSpeed Stage            | 2             |
| DeepSpeed Offloading       | True          |

<details>
<summary>Detailed config</summary>

### General

| Field                      | Value |
|----------------------------|-------|
| save_safetensors           | True  |
| use_gradient_checkpointing | True  |
| trainer_key                | lm    |
| force_fp16                 | False |
| from_gptq                  | False |
| deepspeed_stage            | 2     |
| fsdp_strategy              |       |
| seed                       | 42    |
| stabilize                  | True  |

### Dataset

| Field                    | Value         |
|--------------------------|---------------|
| dataset_key              | soda          |
| train_local_path_to_data | ./train.jsonl |
| eval_local_path_to_data  | None          |
| shuffle                  | True          |

### Tokenizer

| Field                  | Value |
|------------------------|-------|
| tokenizer_name_or_path | None  |
| tokenizer_use_fast     | None  |
| tokenizer_padding_side | right |

### Collator

| Field        | Value |
|--------------|-------|
| collator_key | lm    |
| max_length   | 2048  |

### Model

| Field                 | Value                    |
|-----------------------|--------------------------|
| model_name_or_path    | meta-llama/Llama-2-7b-hf |
| model_type            | llama                    |
| use_flash_attention_2 | True                     |
| trust_remote_code     | True                     |
| device_map            | None                     |

### bitsandbytes

| Field                          | Value |
|--------------------------------|-------|
| model_name_or_pathload_in_8bit | False |
| load_in_4bit                   | True  |
| llm_int8_threshold             | 6.0   |
| llm_int8_has_fp16_weight       | True  |
| bnb_4bit_use_double_quant      | True  |
| bnb_4bit_quant_type            | nf4   |

### Training Arguments

| Field                       | Value      |
|-----------------------------|------------|
| output_dir                  | ./outputs/ |
| per_device_train_batch_size | 4          |
| gradient_accumulation_steps | 4          |
| warmup_steps                | 500        |
| max_steps                   | None       |
| num_train_epochs            | 1          |
| learning_rate               | 2e-4       |
| max_grad_norm               | 1.0        |
| weight_decay                | 0.001      |
| label_smoothing_factor      | 0.1        |
| logging_steps               | 10         |
| save_steps                  | 100        |
| save_total_limit            | 1          |
| push_to_hub                 | True       |

### W&B

| Field           | Value |
|-----------------|-------|
| report_to_wandb | True  |

### LoRA

| Field               | Value |
|---------------------|-------|
| apply_lora          | True  |
| lora_rank           | 64    |
| lora_alpha          | 32    |
| lora_dropout        | 0.1   |
| lora_target_modules | all   |

</details>

## Loss dynamic

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6074d5f1134c000d1ae10d42/QJiPgfDmdQvo1ucWkedOr.png)

---

# 📁 License Details

Please note that Llama 2 has its own
license ([link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)), and the SODA dataset has its own
license as well ([link](https://creativecommons.org/licenses/by/4.0/)). The responsibility for using this model remains
with you.

---

# 🔐 Limitations

The model was trained on a synthetic dataset generated using ChatGPT, leading to a few critical issues with the current
version. Often, the model tends to be rather bland and can occasionally be unnatural. Although the model wasn't
explicitly trained to be safe, it's likely these traits are inherited from ChatGPT. Moreover, handling very long
dialogues is considered out-of-domain for the model since it was trained with a maximum length of 2048 tokens. The
model's ability to generate truth-valid facts wasn't tested, but it's probable that its performance in this area lags
behind OpenAI models. Also, this model wasn't explicitly trained to follow instructions.

---

# 🕹 Use cases

It is suggested to set a maximum context length, for example, 10 messages. Then, store the context in some form of data
storage, such as a database. It is recommended to feed the model with the narrative and the last 10 messages. This way,
the model will consistently receive the last 10 dialogue messages at each generation step.

```python
def generate(prompt: str) -> str:
  ...


max_context_length = 10

narrative = "..."

separator = "\n"

bot_prompt = "Bot"
user_prompt = "Person"

context = list()

while True:
  user_phrase = input("You: ")
  context.append(f"{user_prompt}: {user_phrase}")
  model_prompt = separator.join(
    [narrative] + context[-max_context_length:] + [f"{bot_prompt}:"]
  )
  generated_response = generate(model_prompt)
  bot_phrase = f"{bot_prompt}: {generated_response}"
  context.append(bot_phrase)
  print(bot_phrase)
```

---

# 📋 Dialog examples

<details>
<summary>Example #1</summary>

**Narrative:**
Meilani is a loyal friend who can always be counted on. Meilani is someone you can confide in and trust to keep your
secrets safe. Meilani is always there for a friend in need.

**Context:**
Meilani: So, what's up?
Sarah: Not much. Just feeling a little down lately.
Meilani: That's tough. I'm here for you, though. You can talk to me about anything.
Sarah:

**Original response:**
Thanks, Addy. I appreciate that.

**Shurale responses:**

- 1
- 1
- 1
- 1
- 1

</details>

<details>
<summary>Example #2</summary>

**Narrative:**
Kadee is respected by other people because she is a leader. She is confident and knows what she wants. People admire her
for her strength and determination

**Context:**
Kadee: Hi, how are you?
Person: I'm good, thanks. How about you?
Kadee: I'm doing well. Thanks for asking.
Person: So, what's been going on?
Kadee: Just the usual, work and stuff. You know how it is.
Person:

**Original response:**
Yeah, I can imagine. So, what's new with you?

**Shurale responses:**

- 1
- 1
- 1
- 1
- 1

</details>

<details>
<summary>Example #3</summary>

**Narrative:**
Gabrielle was interested in hiking. She had always wanted to try it and had been told by many people that it was a great
way to get some exercise while also enjoying the outdoors. Finally, she took the plunge and went on a hike with a
friend.

**Context:**
Gabrielle:

**Original response:**
Hey, thanks for coming with me on this hike! I've been wanting to try it for a while.

**Shurale responses:**

- 1
- 1
- 1
- 1
- 1

</details>

<details>
<summary>Example #4</summary>

**Narrative:**
Elizah is obsessed with video games. She spends most of her free time playing them and she loves learning about
different game developers, their processes, and the gaming industry as a whole. Her favorite games are puzzle games, but
she enjoys playing all kinds of games.

**Context:**
Elizah: Hi, Alex! I'm so happy to see you. What have you been up to lately?
Alex: I've been good, thanks for asking. I've just been really busy with work and haven't had much time for anything
else. But I'm glad to have some free time now so we can catch up. So, what's new with you?
Elizah: Oh, not much. I've just been playing a lot of video games lately. I'm really into puzzle games right now, but I
enjoy playing all kinds of games.
Alex: That sounds like a lot of fun. I used to play video games all the time when I was younger, but I don't have as
much time for them now. But it's good to hear that you're enjoying them.
Elizah:

**Original response:**
Yeah, I am. I love learning about different game developers and their processes. And the gaming industry as a whole is
really interesting to me.

**Shurale responses:**

- 1
- 1
- 1
- 1
- 1

</details>

<details>
<summary>Example #5</summary>

**Narrative:**
Simran is a strong and independent woman. She quit her job at Myriah's nightclub because she didn't want to be
associated with that type of business anymore. She wants to focus on her own goals and bettering herself.

**Context:**
Simran: I'm done working at your club, Myriah.
Myriah: What? Why? You're one of my best workers.
Simran: I don't want to be associated with that type of business anymore. It's not what I'm about.
Myriah:

**Original response:**
Fine. If that's how you feel, then I won't force you to stay. But I hope you'll reconsider. We could really use someone
like you at the club.

**Shurale responses:**

- 1
- 1
- 1
- 1
- 1

</details>

---

# 🔮 Benchmark

Coming soon... (maybe will be in V2)

---

# 🛰 Future work

If this model proves successful, I plan to implement an algorithm similar to DeepMind's
ReST ([link](https://arxiv.org/pdf/2308.08998.pdf)). The mentioned work has great potential but has a number of
shortcomings, which I've managed to address in my approach.

---

# 🚀 Call to action

If this model has captured your interest, I urge you to extend your support by liking both the model and
the [X—LLM](https://github.com/KompleteAI/xllm) library, which played an instrumental role in its development.

Are you seeking a dynamic addition to your team who possesses the prowess and the know-how to train such innovative
models? Then consider
sharing [my CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing)
or [LinkedIn](https://www.linkedin.com/in/boriszubarev/) with your manager.

And if your team is hunting for the insights of an adept advisor to propel your projects forward, don't hesitate to
reach out through this website: https://komplete.framer.ai
