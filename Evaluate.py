"""
This script is used to evaluate Aster with chosen dataset and target model
"""
import pickle
import argparse

import numpy as np
import torch
from numpy import linalg as LA
from sklearn.metrics import precision_score, recall_score
from sklearn.cluster import SpectralClustering

from data_preprocessing import data_reader
from target_model_data import Target_Model_pred_fn
from target_model_data import fn_R_given_Selected
from computation_utils import fn_Sample_Generator
from computation_utils import fn_Jacobian_Calculation


np.random.seed(seed=14)
torch.manual_seed(14)

datasets = ["mnist"]
# datasets = ["purchase_2", "purchase_10", "purchase_20", "purchase_50",]
target_models = ["LR"]
# target_models = ["RF", "NN", "LR", "SVM", "DT"]

parser = argparse.ArgumentParser()
parser.add_argument('--n_sample', type=int, default=5000)
parser.add_argument('--n_attack', type=int, default=50)
parser.add_argument('--seed', type=int, default=140)
parser.add_argument('--neighbors', type=int, default=40)
parser.add_argument('--data_generate', type=bool, default=False)
attack_args = parser.parse_args()

# print(attack_args.neighbors)

precisions = []
recalls = []
f1_scores = []

for dataset in datasets:
    for model in target_models:
        np.random.seed(seed=attack_args.seed)
        torch.manual_seed(attack_args.seed)
        filename = dataset + "_" + model + ".pkl"
        print(filename)
        # load data
        orig_dataset, oh_dataset, OH_Encoder = data_reader(dataset)

        class_label_for_count = np.unique(np.hstack([orig_dataset["Y_train"], orig_dataset["Y_test"]]))

        n_class = len(class_label_for_count)
        n_features = orig_dataset['X_train'].shape[1]
        Target_Model = None
        # load pretrained target model
        with open('target_models/' + filename, 'rb') as f:
            Target_Model = pickle.load(f)
        y_attack = np.hstack(([np.ones(int(attack_args.n_attack/2)), np.zeros(int(attack_args.n_attack/2))]))
        x_attack = np.zeros((int(attack_args.n_attack), n_features))
        Jacobian_matrix = np.zeros([attack_args.n_attack, n_class, n_features])

        if attack_args.data_generate:
            output_x = np.zeros((attack_args.n_attack, n_features))
            output_y = y_attack
            classes = np.zeros((attack_args.n_attack, 1))

        for ii in range(attack_args.n_attack):
            R_x, R_y = fn_R_given_Selected(orig_dataset, IN_or_OUT=y_attack[ii])
            R_x_OH = OH_Encoder.transform(R_x.reshape(1, -1))
            x_attack[ii] = R_x
            local_samples = fn_Sample_Generator(R_x, dataset)
            oh_local_samples = OH_Encoder.transform(local_samples)
            local_proba = Target_Model_pred_fn(Target_Model, oh_local_samples)
            R_local_proba = Target_Model_pred_fn(Target_Model, R_x_OH)
            Jacobian_matrix[ii] = fn_Jacobian_Calculation(R_local_proba[0], local_proba, n_features, n_class)

            if attack_args.data_generate:
                output_x[ii] = R_x
                classes[ii] = R_y

        if attack_args.data_generate:
            np.save(f'data/test_data/{dataset}_{model}_x.npy', output_x)
            np.save(f'data/test_data/{dataset}_{model}_y.npy', output_y)
            np.save(f'data/test_data/{dataset}_{model}_class.npy', classes)
        
        Jacobian_norms = LA.norm(Jacobian_matrix, axis=(1, 2))
        # ====================================================================
        split = 1
        attack_cluster = SpectralClustering(n_clusters=6, n_jobs=-1, affinity='nearest_neighbors', n_neighbors=19)
        y_attack_pred = attack_cluster.fit_predict(Jacobian_norms.reshape(-1, 1))
        cluster_1 = np.where(y_attack_pred >= split)[0]
        cluster_0 = np.where(y_attack_pred < split)[0]
        y_attack_pred[cluster_1] = 1
        y_attack_pred[cluster_0] = 0
        cluster_1_mean_norm = Jacobian_norms[cluster_1].mean()
        cluster_0_mean_norm = Jacobian_norms[cluster_0].mean()
        if cluster_1_mean_norm > cluster_0_mean_norm:
            y_attack_pred = np.abs(y_attack_pred-1)
        # ====================================================================
        precision = precision_score(y_attack, y_attack_pred)
        recall = recall_score(y_attack, y_attack_pred)
        f1_score = 2*precision*recall/(precision+recall)
        print(precision, recall, f1_score)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

print("average")
print(sum(precisions)/len(precisions), sum(recalls)/len(recalls), sum(f1_scores)/len(f1_scores))
