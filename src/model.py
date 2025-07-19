import torch
import torch.nn as nn
import numpy as np
import timm

model_dict = {
    'resnet18': [lambda: nn.Sequential(timm.create_model('resnet18', pretrained=True, num_classes=0), nn.Flatten()), 512],
    'resnet34': [lambda: nn.Sequential(timm.create_model('resnet34', pretrained=True, num_classes=0), nn.Flatten()), 512],
    'resnet50': [lambda: nn.Sequential(timm.create_model('resnet50', pretrained=True, num_classes=0), nn.Flatten()), 2048],
    'efficientnetv2': [lambda: nn.Sequential(timm.create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=0), nn.Flatten()), 1280],
    'vit_b_16': [lambda: nn.Sequential(timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0), nn.Flatten()), 768],
    'convnext_b': [lambda: nn.Sequential(timm.create_model('convnext_base', pretrained=True, num_classes=0), nn.Flatten()), 1024],
}
    
class Model(nn.Module):
    def __init__(self, model_backbone, num_classes, num_super_classes):
        super(Model, self).__init__()
        self.model_backbone = model_backbone
        self.num_classes = num_classes
        self.num_super_classes = num_super_classes

        model_fun, dim_in = model_dict[self.model_backbone]
        self.encoder = model_fun()
        self.head_weak = nn.Linear(dim_in, self.num_super_classes)
        self.head_full = nn.Linear(dim_in, self.num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        logits_weak, logits_full = self.head_weak(feat), self.head_full(feat)
        return feat, logits_weak, logits_full
    
    def init(self):
        return Model(self.model_backbone, self.num_classes, self.num_super_classes)

def set_model(model_backbone, num_classes, num_super_classes):
    return Model(model_backbone, num_classes, num_super_classes)

def set_criterion(y_true, num_classes, device):
    N = np.eye(num_classes)[y_true].sum(0)
    class_weights = N.sum() / (N*num_classes+1e-14)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

def set_optimizer(model, optimizer, lr):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optimizer

if __name__ == '__main__':
    model = set_model(model_backbone='resnet18', num_classes=100, num_super_classes=20)
    optimizer = set_optimizer(model=model, optimizer='adam', lr=0.001)

    print(model)
    print(optimizer)