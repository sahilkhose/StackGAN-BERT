'''
Generates bert embeddings from annotations.
'''

import config

import os
import torch
import logging
logging.basicConfig(level=logging.ERROR)

# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

print("__"*80)
print("Imports finished.")
print("Loading BERT Tokenizer and model...")
###############################################################################################
# tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH, do_lower_case=True)
# model = BertModel.from_pretrained(
#     config.BERT_PATH, output_hidden_states=True).to(config.DEVICE)


  
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True).to(config.DEVICE)

model.eval()

###############################################################################################
print("BERT tokenizer and model loaded.")

def sent_emb(sent):
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=128,  # This is changed.
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    with torch.no_grad():
        ### token embeddings:
        outputs = model(input_ids.to(config.DEVICE),
                        attention_masks.to(config.DEVICE))
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)

        ### sentence embeddings:
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.view(1, -1)
        return sentence_embedding


def max_len():
    max_len = 0
    emb_lens = []
    for bird_type in tqdm(sorted(os.listdir(config.ANNOTATIONS)), total=len(os.listdir(config.ANNOTATIONS))):
        for file in sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type))):
            text = open(os.path.join(config.ANNOTATIONS, bird_type, file), "r").read().split('\n')[:-1]
            for annotation in text:
                input_ids = tokenizer.encode(annotation, add_special_tokens=True)
                r = len(input_ids)
                max_len = max(max_len, r)
                emb_lens.append(r)
    print(f"Saving emb lens to {config.EMB_LEN}")
    file = open(config.EMB_LEN, "w")
    for ele in emb_lens:
        file.write(str(ele) + "\n")
    return max_len

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def generate_text_embs():
    print(f"Saving text embeddings to {config.ANNOTATION_EMB}")
    try:
        make_dir(config.ANNOTATION_EMB)
        for bird_type in tqdm(sorted(os.listdir(config.ANNOTATIONS)), total=len(os.listdir(config.ANNOTATIONS))):
            make_dir(os.path.join(config.ANNOTATION_EMB, bird_type))
            for file in sorted(os.listdir(os.path.join(config.ANNOTATIONS, bird_type))):
                make_dir(os.path.join(config.ANNOTATION_EMB, bird_type, file.replace(".txt", "")))
                text = open(os.path.join(config.ANNOTATIONS, bird_type, file), "r").read().split('\n')[:-1]
                for annotation_id, annotation in enumerate(text):
                    emb = sent_emb(annotation)
                    torch.save(emb, os.path.join(config.ANNOTATION_EMB, bird_type, file.replace(".txt", ""), str(annotation_id) + ".pt"))
    except Exception as e:
        print(f"Error in {bird_type}/{file}")
        print(e)


if __name__ == "__main__":
    # To determine mex_length in tokenizer.encode_plus()
    # print("Max length of annotations: ", max_len())  # 80
    
    # Generate sentence embeddings:
    if os.path.exists(config.ANNOTATION_EMB) and len(os.listdir(config.ANNOTATION_EMB)) == len(os.listdir(config.IMAGE_DIR)):  # checking if we have all the embeddings folders created already
        print("Bert embeddings already exist. Skipping...")
    else:
        print("Generating bert embs...")
        generate_text_embs()