import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import copy

from model import set_optimizer, set_criterion
from loader import set_train_loader
from utils import AverageMeter

def train(
    X_train,
    y_train,
    annotation_type,
    val_loader,
    model,
    num_classes, 
    num_super_classes,
    optimizer,
    lr,
    image_size,
    batch_size,
    num_workers,
    num_epochs,
    device
):    
    # ---- training by weak supervision ----
    optimizer = set_optimizer(model=model, optimizer=optimizer, lr=lr)
    criterion_weak = set_criterion(y_true=y_train['weak'][np.arange(len(X_train))[annotation_type!=0]], 
                              num_classes=num_super_classes, device=device)
    train_loader = set_train_loader(X_train, y_train, np.arange(len(X_train))[annotation_type!=0],
                                    image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    # -----
    history_train_loss, history_train_acc_full = [], []
    best_val_acc = 0.0
    train_loss, val_loss = AverageMeter(), AverageMeter()

    for epoch in tqdm(range(num_epochs), leave=False, ncols=50):
        # ---- Train ----
        model.train()
        pred, gt = [], []
        for batch in train_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_weak = batch['y_weak'].to(device, dtype=torch.long)
            n = X.size(0)

            _, logits_weak, _ = model(X)
            loss = criterion_weak(logits_weak, y_weak)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss.item(), n=n)
            pred.extend(logits_weak.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_weak.detach().cpu().numpy())
        
        history_train_loss.append(train_loss.avg)
        gt, pred = np.array(gt), np.array(pred)
        train_acc = accuracy_score(gt, pred)
        history_train_acc_full.append(train_acc*100)
        
        train_loss.reset()

        # ---- Validation ----
        model.eval()
        pred, gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device, dtype=torch.float32)
                y_weak = batch['y_weak'].to(device, dtype=torch.long)
                n = X.size(0)

                _, logits_weak, _ = model(X)
                loss = criterion_weak(logits_weak, y_weak)

                val_loss.update(loss.item(), n=n)
                pred.extend(logits_weak.argmax(dim=-1).detach().cpu().numpy())
                gt.extend(y_weak.detach().cpu().numpy())

        gt, pred = np.array(gt), np.array(pred)
        val_acc = accuracy_score(gt, pred)

        if val_acc*100 >= best_val_acc:
            best_val_acc = val_acc*100
            best_model_param = copy.deepcopy(model.state_dict())

        val_loss.reset()
    
    model.load_state_dict(best_model_param)

    # ---- training by full supervision ----
    optimizer = set_optimizer(model=model, optimizer=optimizer, lr=lr)
    criterion_full = set_criterion(y_true=y_train['full'][np.arange(len(X_train))[annotation_type==2]], 
                                   num_classes=num_classes, device=device)
    train_loader = set_train_loader(X_train, y_train, np.arange(len(X_train))[annotation_type==2],
                                    image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    # ----
    history_train_loss, history_train_acc_full = [], []
    best_val_acc = 0.0
    train_loss, val_loss = AverageMeter(), AverageMeter()

    for epoch in tqdm(range(num_epochs), leave=False, ncols=50):
        # ---- Train ----
        model.train()
        pred, gt = [], []
        for batch in train_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_full = batch['y_full'].to(device, dtype=torch.long)
            n = X.size(0)

            _, _, logits_full = model(X)
            loss = criterion_full(logits_full, y_full)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss.item(), n=n)
            pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_full.detach().cpu().numpy())
        
        history_train_loss.append(train_loss.avg)
        gt, pred = np.array(gt), np.array(pred)
        train_acc = accuracy_score(gt, pred)
        history_train_acc_full.append(train_acc*100)
        
        train_loss.reset()

        # ---- Validation ----
        model.eval()
        pred, gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device, dtype=torch.float32)
                y_full = batch['y_full'].to(device, dtype=torch.long)
                n = X.size(0)

                _, _, logits_full = model(X)
                loss = criterion_full(logits_full, y_full)

                val_loss.update(loss.item(), n=n)
                pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
                gt.extend(y_full.detach().cpu().numpy())

        gt, pred = np.array(gt), np.array(pred)
        val_acc = accuracy_score(gt, pred)

        if val_acc*100 >= best_val_acc:
            best_val_acc = val_acc*100
            best_model_param = copy.deepcopy(model.state_dict())

        val_loss.reset()

    model.load_state_dict(best_model_param)
    return model, best_val_acc

def test(
    model,
    test_loader,
    device
):
    
    model.eval()
    pred, gt = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device, dtype=torch.float32)
            y_full = batch['y_full'].to(device, dtype=torch.long)

            _, _, logits_full = model(X)
            pred.extend(logits_full.argmax(dim=-1).detach().cpu().numpy())
            gt.extend(y_full.detach().cpu().numpy())

    gt, pred = np.array(gt), np.array(pred)
    test_acc = accuracy_score(gt, pred)*100
    
    return test_acc