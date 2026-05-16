import torch
import torch_geometric
import os

pt_path = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG\TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1.pt"

try:
    data = torch.load(pt_path, map_location='cpu')
    print(f"Graph x shape: {data.x.shape}")
    print(f"Graph edge_index shape: {data.edge_index.shape}")
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print(f"Graph edge_attr shape: {data.edge_attr.shape}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
except Exception as e:
    print(f"Error: {e}")
