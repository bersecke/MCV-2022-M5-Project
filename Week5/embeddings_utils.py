from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
from sklearn.neighbors import KNeighborsClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"

mit_classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain',
                         'Opencountry', 'street', 'tallbuilding']
colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728',
            '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f',]

def plot_embeddings(embeddings, targets, legend_cls, colors, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(legend_cls)


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

def extract_dict_retrieval(dataloader, model, size):
    with torch.no_grad():
        model.eval()
        all_ = []
        k = 0
        for images, target, paths in dataloader:
            images = images.to(device)
            embs_aux = model.get_embedding(images).data.cpu().numpy()
            for ind in range(len(images)):
                all_.append({'path': paths[ind], 'emb': embs_aux[ind], 'label': target[ind]})
            k += len(images)
    return all_


def getDistances(comparisonMethod, baseImageHistograms, queryImageHistogram):
    # loop over the index
    results = {}
    for path,label,hist in baseImageHistograms:
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        query = cv2.UMat(np.array(queryImageHistogram, dtype=np.float32))
        histBase = cv2.UMat(np.array(hist, dtype=np.float32))
        distance = cv2.compareHist(query, histBase, comparisonMethod)

        results[path] = (label, distance)
    return results


def build_knn(train_data, train_labels, n, distance='euclidean'):
  knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1,metric=distance)
  knn.fit(train_data, train_labels)
  return knn