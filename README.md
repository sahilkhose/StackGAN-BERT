# ganctober



## Dataset
- Download the The Caltech-UCSD Birds-200-2011 (CUB) Dataset from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and extract it in <b>input/</b> to create <b>input/CUB_200_2011</b> directory which contains the images we need for our task.<br> 
- Read the README about the dataset on the webiste
- Download the text descriptions from: https://drive.google.com/open?id=0B3y_msrWZaXLT1BZdVdycDY5TEE and extract it in <b>input/</b> to create <b>input/birds</b> directory which contains <b>text_c10</b> directory which contains all the annotations needed for our task.<br>
- Download bert_base_uncased from: https://www.kaggle.com/abhishek/bert-base-uncased and extract it in <b>input/</b> to create <b>input/bert_base_uncased</b> to create annotation bert embeddings 
```
ganctober
│   README.md
│   LICENSE   
│
└──>input
│   │   file011.txt
│   │
│   │
│   └──>bert_base_uncased
|   |
|   |
│   │
│   └──>birds
│   |   │   file111.txt
│   |   │   file112.txt
│   |   │   ...
│   |    
|   └──>CUB_200_2011
|
└──>models
|   │   file021.txt
|   │   file022.txt
|
|
└──>src
│   │   file011.txt
│   │   
```