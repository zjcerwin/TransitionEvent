# Transition-based Event Extraction
Extracting Entities and Events as a Single Task Using a Transition-Based Neural Model, code for IJCAI 2019 [paper](https://www.ijcai.org/proceedings/2019/753).

## Dependencies:
+ DyNET 2.1
+ Pyyaml 5.1
+ gensim 3.7.3
+ Pytorch 1.1.0
+ pytorch-transformer 1.2.0
+ flair 0.4.3


## Dataset:
ACE2005 https://catalog.ldc.upenn.edu/ldc2006t06

We can not provide full ACE2005 data files due to LDC license, instead, sample JSON files are given in `data_files/samples/` for reference.

To facilitate follow up research, we list documents split of train/dev/test in `data_files/doc_split/` which is provided by Thien Huu Nguyen et al [2016]

## Configurations 
* `data_config.yaml` (for locating file paths)  
* `joint_config.yaml` (for parameters tunning)

## Preprocess:
+ Put glove.6B.100d.txt in `data_files/glove_emb/`

+ Then make vocabulary and pickle instances by: 
```
python preprocess.py
```

+ (Optional) Generate BERT Embeddings (bert-base-uncased): 
```
python gen_bert_emb.py
```
Note that if you don\`t use BERT, set `use_sentence_vec` to false in `joint_config.yaml`.

It performs poorly for argument roles without BERT embeddings (around 45% F-scores on test set). If you don`t want use BERT, consider incorporating dependency features as in Dependency-Bridge (Sha et al [2018]).

## Train & Evaluate:
```
python train.py
```

## Reference:
+ Thien Huu Nguyen, Kyunghyun Cho, and Ralph Grishman. Joint Event Extraction via Recurrent Neural Networks, NAACL, 2016
+ Lei Sha, Feng Qian, Baobao Chang, and Zhifang Sui. Jointly extracting event triggers and arguments by dependency-bridge rnn and tensor-based argument interaction, AAAI, 2018

## Citation
```
@inproceedings{ijcai2019-753,
  title     = {Extracting Entities and Events as a Single Task Using a Transition-Based Neural Model},
  author    = {Zhang, Junchi and Qin, Yanxia and Zhang, Yue and Liu, Mengchi and Ji, Donghong},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {5422--5428},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/753},
  url       = {https://doi.org/10.24963/ijcai.2019/753},
}
```