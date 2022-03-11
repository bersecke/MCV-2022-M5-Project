import imp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torchvision import transforms
from modelM3 import ModelM3
from tqdm import tqdm
import wandb

wandb.init(project="test-project", entity="fantastic5")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

TRAIN_DATA_PATH = "MIT_split/train/"
TEST_DATA_PATH = "MIT_split/test/"

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

wandb.config = {
  "learning_rate": LEARNING_RATE,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE,
  "architecture" : "CNN",
}

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5] )
    ])

train_data = ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_dataloader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True) 

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = ModelM3().to(device)
# print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_fn = loss_fn.to(device)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(dataloader, model, loss_fn, optimizer, epoch):
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
    acc = correct / len(dataloader)
    print(f"Training loss: {loss:>7f}, Training accuracy: {acc:>7f}")
    wandb.log({"loss": loss, "epoch": t})
    wandb.log({"accuracy": acc, "epoch": t})



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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, t)
    test(test_dataloader, model, loss_fn)
print("Done!")

wandb.watch(model)