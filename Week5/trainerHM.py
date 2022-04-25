import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from networks import EmbeddingNet

from pytorch_metric_learning import miners, losses

device = "cuda" if torch.cuda.is_available() else "cpu"


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, miner = None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        # if epoch == 60:
        #     miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard")

        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, miner)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())


        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, miner):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = [d.cuda() for d in data]
            # if target is not None:
            #     target = target.cuda()

        optimizer.zero_grad()

        #Infer on embeddings
        adapted_img_embeddings = model.get_embedding_img(data[0])
        adapted_postxt_embeddings = model.get_embedding_text(data[1])
        adapted_negtxt_embeddings = model.get_embedding_text(data[2])

        adapted_txt_embeddings = torch.cat((adapted_postxt_embeddings, adapted_negtxt_embeddings)) #Pos & neg combined
        img_labels = target[0]
        txt_labels = torch.cat((target[1], target[2]))

        hard_triplets = miner(adapted_img_embeddings, img_labels, adapted_txt_embeddings, txt_labels)
        
        loss_outputs = loss_fn(adapted_img_embeddings, img_labels, hard_triplets, adapted_txt_embeddings, txt_labels)

        # # outputs = model(*data)

        # if type(outputs) not in (tuple, list):
        #     outputs = (outputs,)

        # loss_inputs = outputs
        # # if target is not None:
        # #     target = (target,)
        # #     loss_inputs += target        


        # loss_outputs = loss_fn(*loss_inputs)

        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # for metric in metrics:
        #     metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            # for metric in metrics:
            #     message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    # loss_fn = nn.TripletMarginLoss(0.2) ##AUTOMATIZE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with torch.no_grad():
        # for metric in metrics:
        #     metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                # if target is not None:
                #     target = target.cuda()

            # outputs = model(*data)

            # if type(outputs) not in (tuple, list):
            #     outputs = (outputs,)
            # loss_inputs = outputs

            # if target is not None:
            #     target = (target,)
            #     loss_inputs += target

            #Infer on embeddings
            adapted_img_embeddings = model.get_embedding_img(data[0])
            adapted_postxt_embeddings = model.get_embedding_text(data[1])
            adapted_negtxt_embeddings = model.get_embedding_text(data[2])

            adapted_txt_embeddings = torch.cat((adapted_postxt_embeddings, adapted_negtxt_embeddings)) #Pos & neg combined
            img_labels = target[0]
            txt_labels = torch.cat((target[1], target[2]))

            loss_outputs = loss_fn(adapted_img_embeddings, img_labels, None, adapted_txt_embeddings, txt_labels)
            # loss_outputs = loss_fn(*loss_inputs)
            
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            # for metric in metrics:
            #     metric(outputs, target, loss_outputs)

    return val_loss, metrics


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    epoch_loss, correct = 0, 0
    for (X, y) in tqdm(dataloader, desc='Training', leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss = epoch_loss / len(dataloader)
    acc = correct / size
    print(f"Training loss: {loss:>7f}, Training accuracy: {acc:>7f}")
    return loss, acc


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

    return test_loss, correct