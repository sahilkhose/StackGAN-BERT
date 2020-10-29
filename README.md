# ganctober



## Dataset
- Download the following and extract/move them to the mentioned directories so that your workspace is similar to the one in the figure.
- Download the The Caltech-UCSD Birds-200-2011 (CUB) Dataset from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and extract it in <b>input/data/</b> to create <b><u>input/data/CUB_200_2011</u></b> directory which contains <b><u>images</u></b> directory with the images we need for our task.<br> 
- Read the README about the dataset on the webiste
- Download the text descriptions from: https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE and extract it in <b>input/data/</b> to create <b><u>input/data/birds</u></b> directory which contains <b><u>text_c10</u></b> directory which contains all the annotations needed for our task.<br>
- Download bert_base_uncased from: https://www.kaggle.com/abhishek/bert-base-uncased and extract it in <b>input/data/</b> to create <b><u>input/data/bert_base_uncased</u></b> to create annotation bert embeddings 
```
ganctober
│   LICENSE
│   README.md   
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
|       |   dataset_check.py
|       |   setup.py
|
└──>models
|
└──>src
│   │   config.py
|   |   dataset.py
│   │   engine.py
│   │   environment.yml
│   │   layers.py
│   │   model.py
│   │   train.py
│   │   utils.py
```
## Generating BERT embeddings of annotations
```bash
python3 input/src/bert_emb.py  
```
