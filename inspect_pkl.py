import pickle
import os

data_dir = r"d:\Hira\Freelance\imran_sheikh\PathomicFusion\data\TCGA_GBMLGG"
pkl_path = os.path.join(data_dir, "gbmlgg15cv_all_st_0_0_0.pkl")

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        if 'cv_splits' in data:
            splits = data['cv_splits']
            print("Number of CV splits:", len(splits))
            first_key = list(splits.keys())[0]
            split = splits[first_key]
            print(f"\nExample Split ({first_key}) Keys:", list(split.keys()))
            for part in ['train', 'test']:
                if part in split:
                    print(f"\n--- {part.capitalize()} Set ---")
                    p_data = split[part]
                    print("Keys:", list(p_data.keys()))
                    if 'x_path' in p_data:
                        print("Sample x_path:", p_data['x_path'][:3])
                        print("Total samples:", len(p_data['x_path']))
                    if 'y_surv' in p_data:
                        print("Sample y_surv (time):", p_data['y_surv'][:3])
                    if 'y_cens' in p_data:
                        print("Sample y_cens:", p_data['y_cens'][:3])
                    if 'x_omic' in p_data:
                        print("Sample x_omic shape:", p_data['x_omic'].shape)
except Exception as e:
    print("Error:", e)
