import numpy as np
import torch
import torch.nn.functional as F
import copy

from loader import set_unlabeled_loader
from train_nn import train

def query(
    round,
    budget,
    cost_weak,
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
    uncertainty,
    device,
):
    num_data = len(X_train)

    # ------------------------------
    if round==0:
        n_add = int(budget/(1+cost_weak))
        # ---- weak annotation ----
        add_index = np.random.choice(np.arange(num_data)[annotation_type==0], size=n_add, replace=False)
        annotation_type[add_index] = 1
        # ---- full annotation ----
        add_index = np.random.choice(np.arange(num_data)[annotation_type==0], size=n_add, replace=False)
        annotation_type[add_index] = 2
    
    # ------------------------------
    else:
        # ---- Step1. cal value-to-cosrt ratio (VCR) ----
        norm_features, probs_weak, probs_full = get_norm_feat_and_prob(model, 
                                                                X_train, 
                                                                y_train, 
                                                                image_size, 
                                                                batch_size, 
                                                                num_workers, 
                                                                device)

        uncetainties_full = get_uncerties(probs_full, uncertainty=uncertainty)
        uncetainties_weak = get_uncerties(probs_weak, uncertainty=uncertainty)
        uncetainties_full = to_percentile(uncetainties_full)
        uncetainties_weak = to_percentile(uncetainties_weak)
        expected_improvement_full, expected_improvement_weak = get_expected_improvement(X_train=X_train,
                                                                                        y_train=y_train,
                                                                                        annotation_type=annotation_type,
                                                                                        val_loader=val_loader,
                                                                                        model=model,
                                                                                        num_classes=num_classes,
                                                                                        num_super_classes=num_super_classes,
                                                                                        optimizer=optimizer,
                                                                                        lr=lr,
                                                                                        image_size=image_size, 
                                                                                        batch_size=batch_size, 
                                                                                        num_workers=num_workers,
                                                                                        num_epochs=num_epochs,
                                                                                        device=device)
        # expected_improvement_full, eexpected_improvement_weak = 3.0, 1.0

        v_full = torch.tensor(uncetainties_full * expected_improvement_full / 1).unsqueeze(-1)
        features_full =  v_full * norm_features

        v_weak = torch.tensor(uncetainties_weak * expected_improvement_weak / cost_weak).unsqueeze(-1)
        features_weak =  v_weak * norm_features

        values = torch.cat([v_weak, v_full]).squeeze()
        features = torch.cat([features_weak, features_full]).to(device).detach()

        # ---- Step2. batch selection ----
        batch_idx = []
        while budget > 0:
            # print('\rbudget: %d, W:%d/%d (%.2f%%), F:%d/%d (%.2f%%)' % (
            #     budget,
            #     sum(annotation_type==1), num_data, (sum(annotation_type==1)/num_data)*100,
            #     sum(annotation_type==2), num_data, (sum(annotation_type==2)/num_data)*100,
            #     ), end='')

            if len(batch_idx)==0: # initial sample
                weak_labeled_index = np.arange(num_data)[annotation_type==1]
                values[weak_labeled_index] = 0
                full_labeled_index = np.arange(num_data)[annotation_type==2]
                values[full_labeled_index] = 0
                values[full_labeled_index+num_data] = 0
                annotated_data_index = values.argmax().item()
                if annotated_data_index < num_data:
                    batch_idx.append(int(annotated_data_index))
                else: 
                    batch_idx.append(int(annotated_data_index))
                    batch_idx.append(int(annotated_data_index-num_data))

            else:
                dm = torch.cdist(features[batch_idx], features, p=2)
                d = dm.min(dim=0)[0].cpu()
                d[batch_idx] = 0

                weak_labeled_index = np.arange(num_data)[annotation_type==1]
                d[weak_labeled_index] = 0
                full_labeled_index = np.arange(num_data)[annotation_type==2]
                d[full_labeled_index] = 0
                d[full_labeled_index+num_data] = 0
                
                p = (d**2) / (d**2).sum()
                annotated_data_index = np.random.choice(np.arange(num_data*2), size=1, p=p)
                if annotated_data_index < num_data:
                    batch_idx.append(int(annotated_data_index))
                else: 
                    batch_idx.append(int(annotated_data_index))
                    batch_idx.append(int(annotated_data_index-num_data))

            # ---- weak annotation ----
            if annotated_data_index < num_data:
                annotation_type[annotated_data_index] = 1
                budget = budget - cost_weak
            # ---- full annotation ----
            else: 
                annotation_type[annotated_data_index-num_data] = 2
                budget = budget - 1

        del dm
        del features
        torch.cuda.empty_cache()
        # print('')

    return annotation_type

# ------------------------------------
def get_norm_feat_and_prob(
    model, 
    X_train, 
    y_train, 
    image_size, 
    batch_size, 
    num_workers, 
    device
):
    unlabeled_loader = set_unlabeled_loader(X=X_train, 
                                            y=y_train, 
                                            load_index=np.arange(len(X_train)),
                                            image_size=image_size,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            )
    model.eval()
    features, probs_weak, probs_full = [], [], []
    with torch.no_grad():
        for batch in unlabeled_loader:
            X = batch['X'].to(device, dtype=torch.float32)

            feat = model.encoder(X)
            norm_feat = F.normalize(feat, dim=1)
            features.extend(norm_feat.cpu().detach().numpy())

            logits_weak, logits_full = model.head_weak(feat), model.head_full(feat)
            probs_weak.extend(F.softmax(logits_weak, dim=-1).cpu().detach().numpy())
            probs_full.extend(F.softmax(logits_full, dim=-1).cpu().detach().numpy())

    return np.array(features), np.array(probs_weak), np.array(probs_full)

# ------------------------------------
def get_expected_improvement(
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
    device,
):
    num_data = len(X_train)
    index_weak = np.arange(num_data)[annotation_type==1]
    index_full = np.arange(num_data)[annotation_type==2]
    num_trial, num_round = 3, 5

    # ---- expercted_improvement_weak ----
    acc_weak = np.zeros([num_trial, num_round])
    for trial in range(num_trial):
        np.random.shuffle(index_weak)
        for round in range(num_round):
            annotation_type_copy = annotation_type.copy()
            annotation_type_copy[index_weak[(round+1)*len(index_weak)//num_round:]] = 0

            model = copy.deepcopy(model)
            model = model.init().to(device)
            _, val_acc = train(X_train=X_train,
                            y_train=y_train,
                            annotation_type=annotation_type_copy,
                            val_loader=val_loader,
                            model=model,
                            num_classes=num_classes, 
                            num_super_classes=num_super_classes,
                            optimizer=optimizer,
                            lr=lr,
                            image_size=image_size,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            num_epochs=num_epochs,
                            device=device)
            acc_weak[trial][round] = val_acc

            # print('number of weak supervision: %d/%d (%.2f%%)'%(sum(annotation_type_copy==1), num_data, (sum(annotation_type_copy==1)/num_data)*100))
            # print('number of full supervision: %d/%d (%.2f%%)'%(sum(annotation_type_copy==2), num_data, (sum(annotation_type_copy==2)/num_data)*100))
            # print('trial: %d, round: %d, acc: %.2f' % (trial, round, val_acc))
    # ------------------------------------
    
    # ---- expercted_improvement_full ----
    acc_full = np.zeros([num_trial, num_round])
    for trial in range(num_trial):
        np.random.shuffle(index_full)
        for round in range(num_round):
            annotation_type_copy = annotation_type.copy()
            annotation_type_copy[index_full[(round+1)*len(index_full)//num_round:]] = 0

            model = copy.deepcopy(model)
            model = model.init().to(device)
            _, val_acc = train(X_train=X_train,
                            y_train=y_train,
                            annotation_type=annotation_type_copy,
                            val_loader=val_loader,
                            model=model,
                            num_classes=num_classes, 
                            num_super_classes=num_super_classes,
                            optimizer=optimizer,
                            lr=lr,
                            image_size=image_size,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            num_epochs=num_epochs,
                            device=device)
            acc_full[trial][round] = val_acc

            # print('number of weak supervision: %d/%d (%.2f%%)'%(sum(annotation_type_copy==1), num_data, (sum(annotation_type_copy==1)/num_data)*100))
            # print('number of full supervision: %d/%d (%.2f%%)'%(sum(annotation_type_copy==2), num_data, (sum(annotation_type_copy==2)/num_data)*100))
            # print('trial: %d, round: %d, acc: %.2f' % (trial, round, val_acc))
    # ------------------------------------

    # print(acc_weak.mean(axis=0))
    # print(acc_full.mean(axis=0))
    improvement_weak_list = cal_improvement(acc_weak.mean(axis=0))
    improvement_full_list = cal_improvement(acc_full.mean(axis=0))

    N_weak, N_full = sum(annotation_type==1), sum(annotation_type==2)
    expected_improvement_weak = WMA(improvement_weak_list) / (N_weak / num_round)
    expected_improvement_full = WMA(improvement_full_list) / (N_full / num_round)

    if expected_improvement_weak <= 0: 
        if expected_improvement_full <= 0:
            expected_improvement_full, expected_improvement_weak = 1, 1
        else:
            expected_improvement_full, expected_improvement_weak = 1, 0
    else:
        if expected_improvement_full <= 0:
            expected_improvement_full, expected_improvement_weak = 0, 1
        else:
            expected_improvement_full = expected_improvement_full / expected_improvement_weak
            expected_improvement_weak = 1
    # print(expected_improvement_full, expected_improvement_weak)

    return expected_improvement_full, expected_improvement_weak
        
# ------------------------------------
def to_percentile(x):
    rank = np.argsort(np.argsort(x))
    percentile = rank / len(x)
    return percentile

def cal_improvement(x):
    return np.array([x[i+1]-x[i] for i in range(len(x)-1)])

def WMA(seq):
    weight = np.arange(len(seq)) + 1
    wma = np.sum(weight * seq) / weight.sum()
    return wma

# ------------------------------------
def get_uncerties(probs, uncertainty='margin'):
    if uncertainty == 'ent':
        return entropy(probs)
    elif uncertainty == 'max_conf':
        return max_conf(probs)
    elif uncertainty == 'margin':
        return margin(probs)
    else:
        raise NameError("uncertainty {} is not supported".format(uncertainty))
        
def entropy(probs):
    probs = torch.tensor(probs)
    log_probs = torch.log(probs)
    U = -(probs*log_probs).sum(1)
    return np.array(U)

def max_conf(probs):
    probs = torch.tensor(probs)
    U = probs.max(1)[0]
    return -np.array(U) # inverse

def margin(probs):
    probs = torch.tensor(probs)
    probs_sorted, _ = probs.clone().detach().sort(descending=True)
    U = probs_sorted[:, 0] - probs_sorted[:,1]
    return -np.array(U) # inverse
