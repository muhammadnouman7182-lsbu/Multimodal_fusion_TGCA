import torch
import pickle
import numpy as np
import os

data_dir = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG"

def inspect_pt(filename):
    print(f"\n--- Inspecting {filename} ---")
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    data = torch.load(path, map_location='cpu')
    print("Type:", type(data))
    if hasattr(data, 'x'):
        print("Node features shape (data.x):", data.x.shape)
    if hasattr(data, 'edge_index'):
        print("Edge index shape (data.edge_index):", data.edge_index.shape)
    if hasattr(data, 'edge_attr'):
        print("Edge features shape (data.edge_attr):", data.edge_attr.shape)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: Tensor of shape {v.shape}")
            else:
                print(f"{k}: {type(v)}")

def inspect_pkl(filename):
    print(f"\n--- Inspecting {filename} ---")
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Keys:", data.keys())
        if 'cv_splits' in data:
            splits = data['cv_splits']
            print("Number of splits:", len(splits))
            first_split_key = list(splits.keys())[0]
            print(f"First split ({first_split_key}) keys:", splits[first_split_key].keys())
            for mode in ['train', 'test']:
                if mode in splits[first_split_key]:
                    m_data = splits[first_split_key][mode]
                    print(f"  {mode} keys:", m_data.keys())
                    if 'x_path' in m_data:
                        print(f"  {mode} x_path sample:", m_data['x_path'][:2])
                        print(f"  {mode} x_path count:", len(m_data['x_path']))

if __name__ == "__main__":
    # Check for the files listed by list_dir
    files = os.listdir(data_dir)
    pt_files = [f for f in files if f.endswith('.pt')]
    for pt in pt_files:
        inspect_pt(pt)
    
    pkl_files = [f for f in files if f.endswith('.pkl')]
    for pkl in pkl_files:
        inspect_pkl(pkl)
