# import transformers

########################################################################

# BERT_PATH = "../input/bert_base_uncased"
# TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

########################################################################

DEVICE = "cuda"
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
MODEL_PATH = "../model/model.bin"
# TRAINING_FILE = "../input/"

# IDS = "../input/ids.txt"
# TEXT_EMB = "../input/text_emb"
# IMG = "../input/img"
# ANNOTATIONS = "../input/annotations"
########################################################################