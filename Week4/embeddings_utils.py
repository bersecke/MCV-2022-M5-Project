import os
import torch
from PIL import Image
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_embeddings(model, img_preprocessor, imgs_path, out_path):
    all_files = []
    for root, dirs, files in os.walk(imgs_path):
        for file in files:
            all_files.append(file)

    embeddings = []
    for file in all_files:
        image = Image.open(file)
        input_tensor = img_preprocessor(image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch.to(device)
        with torch.no_grad():
            output = model(input_batch)
            embeddings.append({'file': file, 'code': output[0]})

    with open(out_path, 'rb') as openFile:
        pickle.dump(embeddings, openFile)