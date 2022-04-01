import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_embeddings(model, img_preprocessor, imgs_path, out_path):
    all_files = []
    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    embeddings = []
    for file in all_files:
        image = Image.open(file)
        input_tensor = img_preprocessor(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch.to(device)
        with torch.no_grad():
            output = model(input_batch)
            embeddings.append({'file': file, 'code': output[0]})
    if not os.path.exists(out_path):
        with open(out_path, 'rb') as openFile:
            pickle.dump(embeddings, openFile)

def extract_embeddings(dataloader, model, size):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), size))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels