# Fine-tuning a Quantized LLM With LoRA (`fineTune_qwen.ipynb` run in Colab)
## Environment setup
### Download LLaMA-Factory
```python
%cd /content/
%rm -rf LLaMA-Factory
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
%ls

!pip install -e .[torch,bitsandbytes]
!pip install -q numpy==2.0.2
```
### Download Qwen2.5-1.5B-Instruct
```python
%cd /content/
!git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```
### Requirement
```python
import json
from pathlib import Path
from google.colab import drive
import random
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

drive.mount('/content/drive')
```
## Processing data
1. Download law data and convert all DISC-Law data into alpaca format 
    ```python
    def read_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def convert_to_alpaca(data, all_data, ref = False):

        for i in range(len(data)):
            otp = data[i]["output"]

            if ref == True:
                inp = "\t".join(data[i]["reference"]) + data[i]["input"] # special case only for Triplet because "ref" exist in the "input" in Triplet-QA
            inp = data[i]["input"]

            item = {
            "instruction": inp,
            "input": "",
            "output": otp
        }
            all_data.append(item)

        return all_data

    # load all data
    %cd /content/drive/MyDrive/Colab Notebooks/fineTune/DISC_LAW
    data_p = read_jsonl("DISC-Law-SFT-Pair.jsonl")
    data_pqa = read_jsonl("DISC-Law-SFT-Pair-QA-released.jsonl")
    data_t = read_jsonl("DISC-Law-SFT-Triplet-released.jsonl")
    data_tqa = read_jsonl("DISC-Law-SFT-Triplet-QA-released.jsonl")

    # prepare data
    all_data = []
    convert_to_alpaca(data_p, all_data, ref = False)
    convert_to_alpaca(data_pqa, all_data, ref = False)
    convert_to_alpaca(data_t, all_data, ref = True)
    convert_to_alpaca(data_tqa, all_data, ref = False)

    random.seed(9)
    random.shuffle(all_data)

    split_ratio = 0.9
    split_point = int(len(all_data) * 0.9)
    train_data = all_data[:split_point]
    test_data = all_data[split_point:]
    print(len(train_data),len(test_data))
    ```
2. Store all data in `law_data.json` in `/LLaMA-Factory/data`, and add data information in the `dataset_info.json`
    ```python
    %cd /content/LLaMA-Factory/data/
    with open("law_data.json", "w", encoding = "utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    data_info = {"law_data":{
                    "file_name": "law_data.json"}
                    }

    dataset_info_path = Path("dataset_info.json")

    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(data_info)

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    ```
## Fine Tune
Modify fine tune parameters
```python
args = dict(
          stage="sft",
          do_train=True,
          model_name_or_path="/content/Qwen2.5-1.5B-Instruct",
          dataset="law_data",
          template="qwen",
          finetuning_type="lora",
          lora_target="all",
          output_dir="/content/LLaMA-Factory/saves/qwen2.5b",
          cutoff_len=512,
          per_device_train_batch_size=10,
          gradient_accumulation_steps=4,
          lr_scheduler_type="cosine",
          logging_steps=100,
          warmup_ratio=0.1,
          save_steps=100,
          learning_rate=5e-5,
          num_train_epochs=2,
          max_samples=12000,  # reduce the size of train_set due to limited online GPU time
          max_grad_norm=1.0,
          loraplus_lr_ratio=16.0,
          fp16=True,
          report_to="none",
          plot_loss=True,
          overwrite_output_dir = True,
          )


%cd /content/LLaMA-Factory/
json.dump(args, open("train_qwen.json", "w", encoding="utf-8"), indent=2)
```
and start training
```python
!llamafactory-cli train train_qwen.json
```
Plot training loss. When the fine tune is finished, training loss plot is saved in `LLaMA-Factory/saves/qwen2.5b`
```python
img = mpimg.imread('/content/LLaMA-Factory/saves/qwen2.5b/training_loss.png')
plt.figure(figsize=(11, 5))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Inferences
```python
%cd /content/LLaMA-Factory/src

from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc


%cd /content/LLaMA-Factory
args = dict(
  model_name_or_path="/content/Qwen2.5-1.5B-Instruct",
  adapter_name_or_path="/content/LLaMA-Factory/saves/qwen2.5b",
  template="qwen",
  finetuning_type="lora",
)
chat_model = ChatModel(args)

# show 5 test data
sampled_test = random.sample(test_data, 5)
test_output = []

for i in range(len(sampled_test)):
  messages = [] #no need for history context

  query = sampled_test[i]["instruction"]
  messages.append({"role": "user", "content": query})
  # print("Assistant: ", end="", flush=True)

  response = ""
  for new_text in chat_model.stream_chat(messages):
    # print(new_text, end="", flush=True)
    response += new_text
  # print()
  messages.append({"role": "assistant", "content": response})

  item = {"question: ": query,
          "generated_answer: ": response,
          "ground_truth: ": sampled_test[i]["output"]}
  # print(item)
  test_output.append(item)

torch_gc()
```
See some output samples:
```python
for data in test_output:
  print(tabulate([(k, v) for k, v in data.items()],
                tablefmt="grid"))
  print("\n\n")
```
## Results and conclusion
### See **Part B** in ``Report_PartA+B.pdf``

