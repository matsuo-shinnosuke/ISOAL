import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from cub200 import load_cub_2011_train_test_as_numpy

def sparse2coarse_cifar100(targets):
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


def download_dataset(dataset='cifar100', budget=1000):
    if dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./dataset/', train=True, download=True)
        test_dataset = datasets.CIFAR100(root='./dataset/', train=False, download=True)

        X_train, X_test = train_dataset.data, test_dataset.data
        y_train, y_test = np.array(train_dataset.targets), np.array(test_dataset.targets)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=budget/len(X_train), random_state=42, stratify=y_train)

        y_train_weak = sparse2coarse_cifar100(y_train)
        y_val_weak = sparse2coarse_cifar100(y_val)
        y_test_weak = sparse2coarse_cifar100(y_test)

    elif dataset == 'cub200':
        X_train, y_train, y_train_weak, X_test, y_test, y_test_weak = load_cub_2011_train_test_as_numpy()

        idx = np.arange(len(y_train))
        idx_train, idx_val = train_test_split(idx, test_size=budget/len(X_train), random_state=42, stratify=y_train)
        X_train, X_val = X_train[idx_train], X_train[idx_val]
        y_train, y_val = y_train[idx_train], y_train[idx_val]
        y_train_weak, y_val_weak = y_train_weak[idx_train], y_train_weak[idx_val]

    else:
        raise NameError("dataset {} is not supported".format(dataset))

    y_train = {'full': y_train, 'weak': y_train_weak}
    y_val = {'full': y_val, 'weak': y_val_weak}
    y_test = {'full': y_test, 'weak': y_test_weak}

    return X_train, y_train, X_val, y_val, X_test, y_test

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, load_index, transform):
        self.X = X
        self.y = y
        self.load_index = load_index
        self.transform = transform

    def __len__(self):
        return len(self.load_index)

    def __getitem__(self, idx):
        idx = self.load_index[idx]

        X = self.X[idx]
        X = self.transform(X)
        y_full, y_weak = self.y['full'][idx], self.y['weak'][idx]
        
        return  {'X': X, 'y_full': y_full, 'y_weak': y_weak, 'idx': idx}
    
def set_train_loader(X, y, load_index, image_size, batch_size=128, num_workers=4):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = Dataset(X=X, y=y, load_index=load_index, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    return train_loader

def set_test_loader(X, y, image_size, batch_size=128, num_workers=4):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = Dataset(X=X, y=y, load_index=np.arange(len(X)), transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers, pin_memory=False)
    return test_loader

def set_unlabeled_loader(X, y, load_index, image_size, batch_size=128, num_workers=4):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    unlabeled_dataset = Dataset(X=X, y=y, load_index=load_index, transform=test_transforms)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return unlabeled_loader
        
if __name__ == '__main__':    
    X_train, y_train, X_val, y_val, X_test, y_test = download_dataset(dataset='cifar100', budget=1000)

    load_index = np.arange(len(X_train))
    train_loader = set_train_loader(X_train, y_train, load_index, image_size=32, batch_size=128, num_workers=4)
    val_loader = set_test_loader(X_val, y_val, image_size=32, batch_size=128, num_workers=4)
    test_loader = set_test_loader(X_test, y_test, image_size=32, batch_size=128, num_workers=4)

    print(len(train_loader), len(val_loader), len(test_loader))
    batch = next(iter(train_loader))
    print(batch['X'].size(), batch['y_full'].size(), batch['y_weak'].size())