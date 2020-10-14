# Utility functions.
# TODO sort data in dirs corresponding to ids
# TODO generate bert embs for annotations, save it

from transformers import BertTokenizer, BertModel
import config
import os
import torch
import logging
logging.basicConfig(level=logging.ERROR)

print("__"*80)
print("Imports finished.")
print("Loading BERT Tokenizer...")
###############################################################################################
tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH, do_lower_case=True)
model = BertModel.from_pretrained(config.BERT_PATH, output_hidden_states=True).to(config.DEVICE)
model.eval()
###############################################################################################

def sent_emb(sent):
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=512, # TODO change this after checking max_len from max_len()
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    with torch.no_grad():
        ### token embeddings:
        outputs = model(input_ids.to(config.DEVICE), attention_masks.to(config.DEVICE))
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)

        ### sentence embeddings:
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.view(1, -1)
        return sentence_embedding

def max_len():
    max_len = 0
    annotations = sorted(os.listdir(config.ANNOTATIONS))
    for annotation in annotations:
        sent = str(open(os.path.join(config.ANNOTATIONS, annotation + ".txt"), "r").read().replace("\n", " "))
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    print("Max length of annotations: ", max_len)
    return max_len 


def generate_text_embs():
    annotations = sorted(os.listdir(config.ANNOTATIONS))
    try:
        for _, annotation in tqdm(enumerate(annotations), total=len(annotations)):
            sent = str(open(os.path.join(config.ANNOTATIONS, annotation + ".txt"), "r").read().replace("\n", " "))
            sent_emb = sent_emb(sent)
            if not os.path.exists(config.TEXT_EMB):
                os.makedirs(config.TEXT_EMB)
            torch.save(sent_emb, os.path.join(config.TEXT_EMB, annotation + ".pt"))
    except:
        print("Error in ", annotation)


if __name__ == "__main__":
    print("Max len of annotations: ", max_len())
    # TODO update the max_len in BERT Model after running this, round up to power of 2
    # generate_text_embs()