import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
from torchvision import transforms

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from tqdm import tqdm
from CaptionDataset import LoaderWrapper



def get_optimzer(initial_lr:float, model:nn.Module):
    optimizer = optim.Adam(params=model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    return optimizer, scheduler




def train_model(optimizer, trainloader:LoaderWrapper, valloader:LoaderWrapper, model:nn.Module, epochs:int = 50, scheduler = None, device = 'cpu', tb_comment = ""):
    running_loss = 0.0
    writer = SummaryWriter(comment=tb_comment)
    for epoch in range(epochs):
        model.train()
        print('Epoch %d' %(epoch+1))
        for i, data in enumerate(tqdm(trainloader()), 0):
            # Get inputs and labels
            inputs, attention_mask, labels = data
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

            # Zero grad
            optimizer.zero_grad()


            # Forward + Backprop
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            #print('Shape:', outputs.shape)

            
            loss = outputs.loss
            loss.backward()
            optimizer.step()


            # Print current stats
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # Get validation accuracy and loss
        #print(labels.shape)
        val_loss = validate_model(model, valloader, device)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Train/Loss", running_loss/(i+1), epoch)

        if scheduler is not None:
            scheduler.step(val_loss)
        running_loss = 0.0
    writer.close()
    return model

def validate_model(model, valloader:LoaderWrapper, device = 'cpu'):
    correct = 0
    total = 0
    total_loss = 0.0
    i = 0
    #model.eval()
    with torch.no_grad():
        for inputs, attention_mask, labels in valloader():
            i += 1   
            # Get inputs and labels
            inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)
            #print(labels.shape)
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

            #print(outputs)
            # Calc loss
            loss = outputs.loss
            total_loss += loss.item()

    return (total_loss/i)

def test_model(model, testloader, transformer_model:bool = False,  device = 'cpu'):
    #model.eval()
    total_preds = None
    total_labels = None
    i = 1
    with torch.no_grad():
        for inputs, labels in testloader:
                
            # Apply image transformations
            # inputs = apply_transforms(transforms, inputs, transformer_model)
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels.shape)
            outputs = model(inputs)

            #if transformer_model:
            #    outputs = outputs.logits

        

            # Predict classes
            _, preds = torch.max(outputs, 1)

            if total_preds is None:
                total_preds = preds.cpu()
                total_labels = labels.cpu()
            else:
                #print('i:', i)
                total_labels = torch.cat((total_labels, labels.cpu())).cpu()
                total_preds = torch.cat((total_preds, preds.cpu())).cpu()

            i+=1
    report = classification_report(y_true=total_labels, y_pred=total_preds)
    print(report)