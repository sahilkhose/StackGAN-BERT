# ganctober

### Dataset
Check README.md in /input
## Generating BERT embeddings of annotations
```bash
$ cd input/src
$ python3 bert_emb.py  
```

### Training
```bash
$ cd src

# To start training
option 1: CLI args training (args.py):
$ python3 train.py --TRAIN_MAX_EPOCH 10 

option 2: yaml args training:
$ python3 train.py --conf ../cfg/s1.yml

$ mkdir ../old_outputs
$ mv ../output ../old_outputs/output_0

$ python3 train.py --conf ../cfg/s2.yml

$ mv ../output ../old_outputs/output_1

# To open tensorboard
$ tensorboard --logdir=../output 
```