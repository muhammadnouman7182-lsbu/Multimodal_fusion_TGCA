import os
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

class PatchDataset(Dataset):
    def __init__(self, patch_list, transform=None):
        self.patch_list = patch_list
        self.transform = transform

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        img_path = self.patch_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

def extract_features(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load UNI model from TIMM
    # Note: MahmoodLab/uni requires huggingface-hub login or local weight path
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Find all slide directories
    slide_dirs = [os.path.join(opt.patch_dir, d) for d in os.listdir(opt.patch_dir) if os.path.isdir(os.path.join(opt.patch_dir, d))]
    
    os.makedirs(opt.output_dir, exist_ok=True)

    for slide_dir in tqdm(slide_dirs, desc="Processing Slides"):
        slide_id = os.path.basename(slide_dir)
        patch_files = [os.path.join(slide_dir, f) for f in os.listdir(slide_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not patch_files:
            continue
            
        dataset = PatchDataset(patch_files, transform=transform)
        loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        
        slide_features = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                features = model(imgs) # [B, 1024]
                slide_features.append(features.cpu())
        
        slide_features = torch.cat(slide_features, dim=0)
        # Average pooling patches to get a slide-level representation (default behavior for pofusion if not using ROI)
        slide_repr = torch.mean(slide_features, dim=0)
        
        torch.save(slide_repr, os.path.join(opt.output_dir, f"{slide_id}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, required=True, help='Directory containing slide-level patch folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted .pt features')
    parser.add_argument('--batch_size', type=int, default=128)
    opt = parser.parse_args()
    
    extract_features(opt)
