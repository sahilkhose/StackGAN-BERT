"""Test a model and generate submission CSV.

Usage:
    > python train.py --load_path PATH --name NAME
    where
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the train run

Authors:
    Abhiraj Tiwari (abhirajtiwari@gmail.com)
    Sahil Khose (sahilkhose18@gmail.com)
"""
import config
import dataset
import engine

import numpy as np
import os
import pandas as pd
import torch


from model import AmazingModel

from sklearn import metrics
from sklearn import model_selection

print("__"*80)
print("Imports Done...")


def run():
    ids = sorted(open(config.IDS, "r").read().split())

    # 60-20-20 split
    id_train, id_test = model_selection.train_test_split(ids, test_size=0.2, shuffle=False, stratify=None)

    id_train, id_valid = model_selection.train_test_split(id_train, test_size=0.25, shuffle=False, stratify=None)

    # Creating DataLoaders
    training_set = dataset.CUBDataset(id_train)
    train_data_loader = torch.utils.data.DataLoader(training_set, batch_size=1, num_workers=1)

    validation_set = dataset.CUBDataset(id_valid)
    valid_data_loader = torch.utils.data.DataLoader(validation_set, batch_size=1, num_workers=1)

    testing_set = dataset.CUBDataset(id_test)
    test_data_loader = torch.utils.data.DataLoader(testing_set, batch_size=1, num_workers=1)

    # Setting up device
    device = torch.device(config.DEVICE)

    # Load model
    load_file = config.MODEL_PATH + "7_model_15.bin"
    model = AmazingModel()
    if os.path.exists(load_file):
        model.load_state_dict(torch.load(load_file))
    model.to(device)

    # Setting up training
    num_train_steps = int(len(id_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_accuracy = 0

    # Main training loop
    for epoch in range(1, config.EPOCHS+1): 
        # Running train, valid, test loop every epoch
        print("__"*80)
        loss = engine.train_fn(train_data_loader, model, optimizer, device, epoch)
        outputs_v, targets_v, loss_v = engine.eval_fn(valid_data_loader, model, device, epoch)
        outputs_t, targets_t, loss_t = engine.eval_fn(test_data_loader, model, device, epoch)

        # Printing losses
        print(f"Epoch {epoch} Training Loss: {loss}")
        print(f"Epoch {epoch} Validation Loss: {loss_v}")
        print(f"Epoch {epoch} Test Loss: {loss_t}\n")

        # Evaluating extra metrics for valid and test
        print("\nVALID:")
        accuracy_v = metrics.accuracy_score(targets_v, torch.max(torch.tensor(outputs_v), 1)[1])
        print(f"Validation Accuracy Score = {accuracy_v}")
        mcc_v = metrics.matthews_corrcoef(targets_v, torch.max(torch.tensor(outputs_v), 1)[1])
        print(f"MCC Score = {mcc_v}")
        cm_v = metrics.confusion_matrix(targets_v, torch.max(torch.tensor(outputs_v), 1)[1])
        print(f"Confusion Matrix: \n {cm_v}")

        print("\nTEST:")
        accuracy_t = metrics.accuracy_score(targets_t, torch.max(torch.tensor(outputs_t), 1)[1])
        print(f"Test Accuracy Score = {accuracy_t}")
        mcc_t = metrics.matthews_corrcoef(targets_t, torch.max(torch.tensor(outputs_t), 1)[1])
        print(f"MCC Score = {mcc_t}")
        cm_t = metrics.confusion_matrix(targets_t, torch.max(torch.tensor(outputs_t), 1)[1])
        print(f"Confusion Matrix: \n {cm_t}")

        # Printing gold standard for first epoch
        if epoch == 1:
            print("\nALL ONES")
            all_ones_acc = metrics.accuracy_score(targets_t, list(np.ones((len(targets_t)))))
            print("ACCURACY: ", all_ones_acc)
            all_ones_mcc = metrics.matthews_corrcoef(targets_t, list(np.ones((len(targets_t)))))
            print(f"MCC Score = {all_ones_mcc}")
            all_ones_cm = metrics.confusion_matrix(targets_t, list(np.ones((len(targets_t)))))
            print(f"Confusion Matrix: \n {all_ones_cm}")

        # Saving checkpoints
        if accuracy_t > best_accuracy:
            print(f"Saving the best model! Test Accuracy: {accuracy_t}, All ones: {all_ones_acc}")
            torch.save(model.state_dict(), config.MODEL_PATH + f"{config.NUM}_model_{epoch}.bin")
            best_accuracy = accuracy_t

        print("__"*80)
        if epoch % 50 == 0:
            print(f"Saving intermediate model! Test Accuracy: {accuracy_t}, All ones: {all_ones_acc}")
            torch.save(model.state_dict(), config.MODEL_PATH + f"{config.NUM}_model_{epoch}.bin")
        if epoch == config.EPOCHS:
            print(f"Saving the last model! Test Accuracy: {accuracy_t}, All ones: {all_ones_acc}")
            torch.save(model.state_dict(), config.MODEL_PATH + f"{config.NUM}_model_{epoch}.bin")


if __name__ == "__main__":
    run()
