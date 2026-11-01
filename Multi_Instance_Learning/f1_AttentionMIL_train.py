import itertools
import yaml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import subprocess
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    WeightedRandomSampler,
)
from torch.utils.tensorboard import SummaryWriter

from model_AttentionMIL import AttentionNet

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def get_feature_bag_path(data_dir, slide_id):
    return os.path.join(data_dir, f"{slide_id}.h5")

class FeatureBagsDataset(Dataset):
    def __init__(self, df, data_dir):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df["label"][idx]

        full_path = get_feature_bag_path(self.data_dir, slide_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]

        features = torch.from_numpy(features)
        return features, label, coords

    def __len__(self):
        return len(self.slide_df)

def evaluate_model(model, loader, n_classes, loss_fn, device):
    model.eval()

    avg_loss = 0.0

    preds = np.zeros(len(loader))
    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data)

            loss = loss_fn(logits, label)
            avg_loss += loss.item()

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

    avg_loss /= len(loader)

    return preds, probs, labels, avg_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, min_epochs=50, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_epochs (int): Earliest epoch possible for stopping.
            verbose (bool): If True, prints messages for e.g. each validation loss improvement.
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def compute_acc(labels, preds):
    assert labels.shape[0] > 0
    assert preds.shape[0] > 0

    return accuracy_score(labels, preds)

def compute_auc_binary(labels, probs):
    assert labels.shape[0] > 0
    assert probs.shape[0] > 0
    assert probs.shape[1] > 1

    return roc_auc_score(labels == 1, probs[:, 1])

def compute_f1_score(labels, preds):
    assert labels.shape[0] > 0
    assert preds.shape[0] > 0

    return f1_score(labels, preds)

def run_train_eval_loop(
        train_loader,
        val_loader,
        input_feature_size,
        class_names,
        hparams,
        overall_id,
        run_id,
        save_checkpoints
):
    writer = SummaryWriter(os.path.join("D:\\Python_project\\Macrosection\\Lymph_Nodes\\Modeling_Results\\", overall_id, run_id))
    device = torch.device("cuda")
    loss_fn = torch.nn.CrossEntropyLoss()
    n_classes = len(class_names)
    model = AttentionNet(
        model_size=hparams["model_size"],
        input_feature_size=input_feature_size,
        dropout=True,
        p_dropout_fc=hparams['p_dropout_fc'],
        p_dropout_atn=hparams['p_dropout_atn'],
        n_classes=n_classes
    )
    model.to(device)
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],
    )

    milestones = [int(x) for x in hparams["milestones"].split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
    )

    early_stop_tracker = EarlyStopping(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        verbose=True,
    )

    metric_history = []
    for epoch in range(hparams['max_epochs']):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        preds = np.zeros(len(train_loader))
        probs = np.zeros((len(train_loader), n_classes))
        labels = np.zeros(len(train_loader))

        batch_start_time = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            data_load_duration = time.time() - batch_start_time

            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data)

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().detach().numpy()
            labels[batch_idx] = label.item()

            loss = loss_fn(logits, label)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            batch_duration = time.time() - batch_start_time
            batch_start_time = time.time()

            print(
                f"epoch {epoch}, batch {batch_idx}, batch took: {batch_duration:.2f}s, data loading: {data_load_duration:.2f}s, loss: {loss.item():.4f}, label: {label.item()}"
            )

        epoch_duration = time.time() - epoch_start_time
        print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

        train_loss /= len(train_loader)
        train_auc = compute_auc_binary(labels, probs)
        train_acc = compute_acc(labels, preds)
        train_f1_score = compute_f1_score(labels, preds)

        writer.add_scalar('LR', get_lr(optimizer), epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar("AUC/train", train_auc, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1 Scores/train", train_f1_score, epoch)

        writer.add_pr_curve(
            f"PR Curve/train",
            labels == 1,
            probs[:, 1],
            epoch,
        )

        preds, probs, labels, val_loss = evaluate_model(
            model, val_loader, n_classes, loss_fn, device
        )
        val_auc = compute_auc_binary(labels, probs)
        val_acc = compute_acc(labels, preds)
        val_f1_score = compute_f1_score(labels, preds)

        writer.add_scalar("Loss/validation", val_loss, epoch)
        writer.add_scalar("AUC/validation", val_auc, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.add_scalar("F1 Scores/validation", val_f1_score, epoch)

        writer.add_pr_curve(
            f"PR Curve/validation",
            labels == 1,
            probs[:, 1],
            epoch,
        )

        metric_dict = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "val_f1_score": val_f1_score,
            "trainable_params": n_trainable_params,
        }

        metric_history.append(metric_dict)
        #early_stop_tracker(epoch, val_loss)

        if save_checkpoints:
            torch.save(
                model.state_dict(),
                os.path.join(writer.log_dir, f"{epoch}_checkpoint.pt"),
            )

        scheduler.step()

        #if early_stop_tracker.early_stop:
            #print(
                #f"Early stop criterion reached. Broke off training loop after epoch {epoch}."
            #)
            #break

    best = sorted(metric_history, key=lambda x: x["val_auc"])[0]
    writer.add_hparams(hparams, best)

    writer.close()

def define_data_sampling(train_split, val_split, method, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif method == "balanced":
        print("balanced sampling setting")
        train_labels = train_split.slide_df["label"]

        # Compute sample weights to alleviate class imbalance with weighted sampling.
        sample_weights = compute_sample_weight("balanced", train_labels)

        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            # Use the weighted sampler using the precomputed sample weights.
            # Note that replacement is true by default, so
            # some slides of rare classes will be sampled multiple times per epoch.
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    # val_split would be an empty list if not validation is asked in training.
    if len(val_split) == 0:
        val_loader = val_split
    else:
        val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return train_loader, val_loader

def get_class_names(df):
    n_classes = len(df["label"].unique())
    class_names = [None] * n_classes
    for i in df["label"].unique():
        class_names[i] = df[df["label"] == i]["class"].unique()[0]
    assert len(class_names) == n_classes
    return class_names

def main_function(args):

    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )

    ####借由args传参
    workers = args['defaults']['workers']
    hparam_options = args['hparams']
    hparam_id = args['customs']['hparams_to_use']
    hparam_id = ["hparams_set" + a for a in hparam_id.split(",")]
    hparam_to_use = [hparam_options[hparam_id[a]] for a in range(0, len(hparam_id), 1)]

    manifest = args['customs']['manifest_file_path']
    data_dir = args['customs']['features_save_path']
    input_feature_size = args['customs']['feature_size']
    train_run_id = args['customs']['train_run_id']

    ####继续主函数
    df = pd.read_excel(manifest)
    class_names = get_class_names(df)

    training_set = df[df["group"] == "training"]
    validation_set = df[df["group"] == "validation"]

    train_split = FeatureBagsDataset(training_set, data_dir)
    val_split = FeatureBagsDataset(validation_set, data_dir)
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    for i, hps in enumerate(hparam_to_use):
        run_id = f"{train_run_id}_{hps['model_size']}_model_{hps['sampling_method']}_{hparam_id[i]}"
        print(hps)

        train_loader, val_loader = define_data_sampling(
            train_split,
            val_split,
            method=hps["sampling_method"],
            workers=workers,
        )

        run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            input_feature_size=input_feature_size,
            class_names=class_names,
            hparams=hps,
            overall_id=train_run_id,
            run_id=run_id,
            save_checkpoints=True,
        )

if __name__ == "__main__":
    ####yaml传参
    with open('D:\\Python_project\\Macrosection\\Lymph_Nodes\\lymph_node_ABMIL_trainval.yaml', 'r') as stream:
        args = yaml.safe_load(stream)
    ####主函数，启动
    main_function(args)