# The Seq2Seq model for translation (`seq2seq_cmn2eng.ipynb` run in Kaggle GPU)
## Requirement
```python
!pip install opencc

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from opencc import OpenCC

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import time
import math
import matplotlib.ticker as ticker
import numpy as np
import time
import json

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

## Processing data
1. Extract English-Chinese pairs  
Download `cmn.txt` from https://www.manythings.org/anki/cmn-eng.zip  
```python
clean_file('cmn.txt', 'eng-cmn.txt')
```
2. Loading data files  
   Similar to [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)  
   Borrow code from the link above, including:
   ```python
   class Lang...
   def readLangs...
   def prepareData...
   ```
   Tokenize English sentence by words, make each word lowercase and remove punctuations.  
   For Chinese sentences, tokenize them by characters, and convert traditional Chinese characters into simplified version. See `seq2seq_cmn2eng.ipynb`
   ```python
   def normalizeEn(s)
   def normalizeCh(s)`
   ```
3. Convert to training data  
   Borrow code from [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
   ```python
   def indexesFromSentence(lang, sentence)
   def tensorFromSentence(lang, sentence)
   def tensorsFromPair(pair)
   def get_dataloader(batch_size, ratio=0.9) ## add spliting training and testing data function
   ```
## The Seq2Seq Model
Borrow code from [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), including:
```python
class EncoderRNN(nn.Module)
class BahdanauAttention(nn.Module)
class AttnDecoderRNN(nn.Module)
```

## Training and evaluate (& visualize attention)
Borrow some codes from [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) with modification, including:
```python
def train_epoch...
def train...
def evaluate...
def showAttention...
def evaluateAndShowAttention...
```

## Main function
```python
# hyperparameters
hidden_size = 256
batch_size = 64

# prepocessing data
clean_file('/kaggle/input/trans-data/cmn.txt', 'eng-cmn.txt')
input_lang, output_lang, train_dataloader, test_data = get_dataloader(batch_size,ratio=0.9)

# seq2seq & training
Encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
Decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
train_loss = train(train_dataloader, Encoder, Decoder, 100, print_every=5)
showPlot(train_loss)

# evaluate
Encoder.eval()
Decoder.eval()
output = evaluateOutput(Encoder, Decoder, test_data, len(test_data))

#visualize attention example
data = output
for i in range(3):
    evaluateAndShowAttention(data[i][0],data[i][2], data[i][1], np.array(data[i][3]))
```
## Results and conclusion
### See **Part A.a** in `Report_PartA+B.pdf`
  
    
# minGPT for translation (`gpt_cmn2eng.ipynb` run in Kaggle GPU)
## Requirement
Install minGPT by 
```python
!pip install git+https://github.com/karpathy/minGPT.git > /dev/null
```
```python
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import pickle
from opencc import OpenCC
import random
from torch.utils.data.dataloader import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata
import json
import pandas as pd

from mingpt.model import GPT
from mingpt.trainer import Trainer
```
## Processing data
Similarly, we need to prepocessing English words and Chinese characters by `normalizeEn(s)` and `normalizeCh(s)`  
Convert data to training data
```python
class TranslationDataset(Dataset):
    def __init__(self, split, filepath = "eng-cmn.txt", block_size=128):
        self.pairs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            idx = 0
            for line in f:
                if '\t' in line:
                    en, zh = line.strip().split('\t')[:2]
                    zh = zh.replace(" ", "")
                    h = hash(pickle.dumps((zh, en)))
                    inp_split = 'test' if h % 10 == 0 else 'train'  # 10% for test dataset
                    if inp_split != split:
                        continue

                    self.pairs.append((normalizeCh(zh), normalizeEn(en)))
                idx += 1
                if (idx+1)%5000 == 0:
                    print(f"processing line {idx+1}")
            print(f"processsing line {len(self.pairs)}. Done.")
        
        # use GPT2Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sep_token = "<sep>"
        self.eos_token = "<|endoftext|>"

        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': [self.sep_token]})

        self.block_size = block_size
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.tokenizer) 
        
        random.seed(9)
        random.shuffle(self.pairs)
        self.split = split

    def __len__(self):
        return len(self.pairs)

    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        return self.block_size

    def __getitem__(self, idx):
        zh, en = self.pairs[idx]

        encoded_zh = self.tokenizer(zh+self.sep_token, return_tensors="pt")["input_ids"].squeeze(0)
        encoded_en = self.tokenizer(en+self.eos_token, return_tensors="pt")["input_ids"].squeeze(0)

        cat = torch.cat((encoded_zh, encoded_en), dim=0)

        x = cat[:-1].clone()
        y = cat[1:].clone()
        sep_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        eos_id = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        y_start = y.tolist().index(sep_id)
        pad_x = torch.ones(self.block_size, dtype=torch.long) * eos_id
        pad_x[:len(x)] = x
        pad_y = torch.ones(self.block_size,dtype=torch.long) * (-1)
        pad_y[y_start+1:y_start+1+len(encoded_en)] = encoded_en

        return  pad_x, pad_y
```
## Training
```python
# prepare dataset
train_dataset = TranslationDataset('train')
test_dataset = TranslationDataset('test')

# set model configuration parameters
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-micro'
model_config.vocab_size = train_dataset.get_vocab_size()
model_config.block_size = train_dataset.get_block_size()
model = GPT(model_config)

# use minGPT trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4
train_config.max_iters = 3000
train_config.num_workers = 0
train_config.batch_size = 64
train_config.n_layer = 4
train_config.weight_decay =0.05
trainer = Trainer(train_config, model, train_dataset)

training_losses = []
def batch_end_callback(trainer):
    if trainer.iter_num % 20 == 0:
        training_losses.append(trainer.loss.item())
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

# plot training losses
plt.figure(figsize = (10,5))
plt.plot(np.arange(len(training_losses))*20, training_losses)
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.grid()
```

## Evaluation
```python
model.eval()
loader = DataLoader(test_dataset, batch_size=1, num_workers=0, drop_last=False)

output = {}

for b,(x, y) in enumerate(loader):

    x = x.to(trainer.device)
    y = y.to(trainer.device)
    # print(x,y)
    sep_id = train_dataset.tokenizer.convert_tokens_to_ids("<sep>")
    sep_idx = x.tolist()[0].index(sep_id)
    inp = x[:,:sep_idx]
    # print(inp)
    sol_can = model.generate(inp, 128, do_sample=False)

    tokens = y.cpu().tolist()[0]
    #ground truth
    gt = []
    for i in tokens:
        if i != -1:
            gt.append(i)
    gt_sentence = test_dataset.tokenizer.decode(gt, skip_special_tokens=True)
    
    with torch.no_grad():
        generate_sol = sol_can.cpu().tolist()[0]    
    decoded_text = test_dataset.tokenizer.decode(generate_sol, skip_special_tokens=True)
    for i in range(len(decoded_text)): 
        if decoded_text[i] in train_dataset.vocab:
            break
    zh = decoded_text[:i]
    translation = decoded_text[i:]

    output[b] = {"input": zh,
                 "ground_truth": gt_sentence,
                 "translation": translation}

    
    if (b+1) % 300 == 0:
        print(f"Input:  {zh}")
        print(f"Ground truth: {gt_sentence}")
        print(f"Translation: {translation}")
        print()
```

## Results and conclusion
### See **Part A.b** in `Report_PartA+B.pdf`