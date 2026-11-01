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
from sklearn.metrics import confusion_matrix, roc_auc_score
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

def get_class_names(df):
    n_classes = len(df["label"].unique())
    class_names = [None] * n_classes
    for i in df["label"].unique():
        class_names[i] = df[df["label"] == i]["class"].unique()[0]
    assert len(class_names) == n_classes
    return class_names

def load_trained_model(device, checkpoint_path, model_size, input_feature_size, n_classes, dropout_fc, dropout_atn):
    model = AttentionNet(input_feature_size=input_feature_size,
                         n_classes=n_classes,
                         model_size=model_size,
                         p_dropout_fc=dropout_fc,
                         p_dropout_atn=dropout_atn)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, loader, n_classes, device):
    preds = np.zeros(len(loader))
    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            logits, Y_prob, Y_hat, _, _ = model(data)

            preds[batch_idx] = Y_hat.item()
            probs[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

    return preds, probs, labels

def define_data_sampling(testing_split, workers):
    g = torch.Generator()
    g.manual_seed(0)

    testing_loader = DataLoader(
        dataset=testing_split,
        batch_size=1,
        sampler=SequentialSampler(testing_split),
        collate_fn=collate,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return testing_loader

def main_function(args):

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ####借由args传参
    workers = args['defaults']['workers']

    train_run_id = args['customs']['train_run_id']
    manifest = args['customs']['manifest_file_path']
    data_dir = args['customs']['features_save_path']
    model_size = args['customs']['model_size']
    input_feature_size = args['customs']['feature_size']
    t_dropout_fc = args['customs']['dropout_fc']
    t_dropout_atn = args['customs']['dropout_atn']

    ####继续主函数
    df = pd.read_excel(manifest)
    class_names = get_class_names(df)

    ####加载测试集
    testing_set = df[df["group"] == "testing"]
    testing_split = FeatureBagsDataset(testing_set, data_dir)
    print(f"=> Testing on {len(testing_split)} samples")

    testing_loader = define_data_sampling(testing_split, workers)
    slide_list = testing_loader.sampler.data_source.slide_df["slide_id"].tolist()

    ####加载模型
    training_result_file = args['customs']['training_result_file']
    training_result_file = pd.read_excel(training_result_file)

    for i in range(0,training_result_file.shape[0]):
        model_name = os.path.join("D:\\Python_project\\Macrosection\\Lymph_Nodes\\Modeling_Results", train_run_id,
                                  training_result_file.iloc[i]["Task_Name"],
                                  str(training_result_file.iloc[i]["Epoch"])+"_checkpoint.pt")
        attn_model = load_trained_model(device, model_name, model_size, input_feature_size, len(class_names),
                                        t_dropout_fc, t_dropout_atn)
        attn_model.to(device)
        preds, probs, labels = evaluate_model(attn_model, testing_loader, len(class_names), device)

        output_results = np.vstack((preds, probs[:, 0], probs[:, 1], labels))
        output_results = output_results.T
        output_results = pd.DataFrame(output_results, index=slide_list,
                                      columns=["pred_label", "prob_1", "prob2", "true_label"])
        output_results.columns.values[1:3] = class_names
        output_results_save_dir = os.path.join("D:\\Python_project\\Macrosection\\Lymph_Nodes\\Test_Results", train_run_id)
        if not os.path.exists(output_results_save_dir):
            os.makedirs(output_results_save_dir)
        output_results.to_excel(os.path.join(output_results_save_dir, training_result_file.iloc[i]["Task_Name"]+"_FJCH.xlsx"))

if __name__ == "__main__":
    ####yaml传参
    with open('D:\\Python_project\\Macrosection\\Lymph_Nodes\\lymph_node_ABMIL_test_FJCH.yaml', 'r') as stream:
        args = yaml.safe_load(stream)
    ####主函数，启动
    main_function(args)