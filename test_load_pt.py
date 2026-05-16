import torch
import os

path = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG\TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1.pt"

print(f"Attempting to load: {path}")
if not os.path.exists(path):
    print("FILE NOT FOUND")
else:
    try:
        # Some .pt files from torch_geometric might need torch_geometric imported
        try:
            import torch_geometric
        except:
            pass
        data = torch.load(path, map_location='cpu')
        print("LOAD SUCCESS")
        print("Type:", type(data))
        if hasattr(data, '__dict__'):
            print("Attributes:", list(data.__dict__.keys()))
        else:
            print("Data:", data)
    except Exception as e:
        print("LOAD FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
