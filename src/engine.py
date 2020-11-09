# import config
import torch
import torch.nn as nn
from tqdm import tqdm

# TODO following is very basic {train, eval}_fn and random loss_fn. Figure out for GANs

def loss_fn(outputs, images):  
    return nn.CrossEntropyLoss(weight=torch.tensor([[0.538, 0.462]]).to(device="cuda"))(outputs, targets.view(-1).long())

def train_fn(data_loader, model, optimizer, device, epoch): 
    model.train()
    LOSS = 0.

    for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train Epoch {epoch}/{config.EPOCHS}"):
        text_embs, images = data

        # Loading it to device
        text_embs = text_embs.to(device, dtype=torch.float)
        images = images.to(device, dtype=torch.float)

        # getting outputs from model and calculating loss
        optimizer.zero_grad()
        outputs = model(text_embs, images)  

        loss = loss_fn(outputs, images)  # TODO figure this out
        LOSS += loss
        loss.backward()
        optimizer.step()

    LOSS /= len(data_loader)
    return LOSS

def eval_fn(data_loader, model, device, epoch):
    model.eval()
    fin_y = []
    fin_outputs = []
    LOSS = 0.

    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            text_embs, images = data

            # Loading it to device
            text_embs = text_embs.to(device, dtype=torch.float)
            images = images.to(device, dtype=torch.float)
            
            # getting outputs from model and calculating loss
            outputs = model(text_embs, images)
            loss = loss_fn(outputs, images) # TODO figure this out
            LOSS += loss

            # for calculating accuracy and other metrics # TODO figure this out
            fin_y.extend(images.view(-1, 1).cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    LOSS /= len(data_loader)
    return fin_outputs, fin_y, LOSS
