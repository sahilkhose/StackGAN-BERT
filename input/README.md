## :four_leaf_clover: New
- run the following to automate the dataset download
```bash
cd input/src
python3 data.py
```
```
ganctober
│   LICENSE
│   README.md   
│	requirements.txt
│
└──>cfg
│
└──>examples
│
└──>input
│   │
│   │
│   └──>data
│   |   │
│   |   └──>bert_base_uncased
|   |   |
│   |   └──>birds
│   |   |    
|   |   └──>CUB_200_2011
│   │
│   └──>src
|       |   bert_emb.py
|       |   config.py
|       |   data.py
|       |   dataset_check.py
|       |   setup.py
|
|
└──>old_outputs
|
└──>output
|
└──>src
│   │   args.py
|   |   dataset.py
│   │   engine.py
│   │   environment.yml
│   │   layers.py
│   │   train.py
│   │   util.py
```
--------------------------------------------------------------------------------------------

## Dataset (Old)
- Download the following and extract/move them to the mentioned directories so that your workspace is similar to the one in the figure.
- Download the The Caltech-UCSD Birds-200-2011 (CUB) Dataset from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and extract it in `input/data/` to create `input/data/CUB_200_2011` directory which contains `images` directory with the images we need for our task.<br> 
- Read the README about the dataset on the webiste
- Download the text descriptions from: https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE and extract it in `input/data/` to create `input/data/birds` directory which contains `text_c10` directory which contains all the annotations needed for our task.<br>
- Download bert_base_uncased from: https://www.kaggle.com/abhishek/bert-base-uncased and extract it in `input/data/` to create `input/data/bert_base_uncased` to create annotation bert embeddings 