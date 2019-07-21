import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(labels), 0)
    mask = np.ones((len(labels), K), dtype=bool)
    mask[range(len(labels)), labels.numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return complementary_labels

def class_prior(complementary_labels):
    return np.bincount(complementary_labels) / len(complementary_labels)

def prepare_mnist_data(batch_size):
    ordinary_train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    full_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=len(ordinary_train_dataset.data), shuffle=True)
    num_classes = len(ordinary_train_dataset.classes)
    return full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes

def prepare_train_loaders(full_train_loader, batch_size, ordinary_train_dataset):
    for i, (data, labels) in enumerate(full_train_loader):
            K = torch.max(labels)+1 # K is number of classes, full_train_loader is full batch
    complementary_labels = generate_compl_labels(labels)
    ccp = class_prior(complementary_labels)
    complementary_dataset = torch.utils.data.TensorDataset(data, torch.from_numpy(complementary_labels).float())
    ordinary_train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset, batch_size=batch_size, shuffle=True)
    complementary_train_loader = torch.utils.data.DataLoader(dataset=complementary_dataset, batch_size=batch_size, shuffle=True)
    return ordinary_train_loader, complementary_train_loader, ccp

