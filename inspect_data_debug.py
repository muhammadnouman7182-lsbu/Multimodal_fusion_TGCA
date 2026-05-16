import torch
import pickle
import numpy as np
import os
import traceback

data_dir = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG"

def inspect_pt(filename):
    print(f"\n--- Inspecting {filename} ---")
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    try:
        # Try loading with torch_geometric if it's a GNN data object
        import torch_geometric
        data = torch.load(path, map_location='cpu')
        print("Type:", type(data))
        if hasattr(data, 'x'):
            print("Node features shape (data.x):", data.x.shape)
        if hasattr(data, 'edge_index'):
            print("Edge index shape (data.edge_index):", data.edge_index.shape)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        traceback.print_exc()

def inspect_pkl(filename):
    print(f"\n--- Inspecting {filename} ---")
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    try:
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
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    files = os.listdir(data_dir)
    for f in files:
        if f.endswith('.pt'):
            inspect_pt(f)
        elif f.endswith('.pkl'):
            inspect_pkl(f)
