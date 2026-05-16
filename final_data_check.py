import pickle
import os
import torch

data_dir = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG"
pkl_path = os.path.join(data_dir, "gbmlgg15cv_all_st_0_0_0.pkl")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

splits = data['cv_splits']
first_key = list(splits.keys())[0]
train_data = splits[first_key]['train']

print(f"Omic shape: {train_data['x_omic'].shape}")
print(f"Number of training samples: {len(train_data['x_path'])}")
print(f"Number of test samples: {len(splits[first_key]['test']['x_path'])}")

pt_path = os.path.join(data_dir, "TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1.pt")
import torch_geometric
graph = torch.load(pt_path, map_location='cpu')
print(f"Graph x shape: {graph.x.shape}")
print(f"Graph edge_index shape: {graph.edge_index.shape}")
