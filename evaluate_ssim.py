import torch
import os

import numpy as np
import torch.nn.functional as F
import pandas as pd

from argparse import ArgumentParser
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
# from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim


class COCODataset(Dataset):
    def __init__(self, csv_file, coco_path, transform=None):
        self.filenames = []
        self.prompts = []
        if not os.path.exists(coco_path):
            coco_path = "/data/datasets/coco/val2014"
        with open(csv_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                filename = line[:29]
                prompt = line[30:].replace("\"", "")
                self.filenames.append(filename)
                self.prompts.append(prompt)
        self.coco_path = coco_path
        self.transform = transform
        assert len(self.filenames) == len(self.prompts)

    def __len__(self):  
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.coco_path, self.filenames[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.prompts[idx], self.filenames[idx]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--fake_path", type=str)
    parser.add_argument("--coco_type", choices=["5k", "30k"], default="5k")

    args = parser.parse_args()
    return args

def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.CenterCrop((512, 512)),
                                    transforms.ToTensor()])
    
    dataset = COCODataset(csv_file=f"coco_val2014_{args.coco_type}.csv", coco_path="/local_datasets/coco/val2014", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # real_image_paths = sorted(os.listdir("/data/datasets/coco/val2014"))
    # real_image_paths = [os.path.join("/data/datasets/coco/val2014", path) for path in real_image_paths]

    fake_image_paths = sorted(os.listdir(args.fake_path))
    fake_image_paths = [os.path.join(args.fake_path, path) for path in fake_image_paths]
    start_time = datetime.now()
    ssim_scores = []

    for i, (real_image, _, file_name) in enumerate(dataloader):
        
        
        fake_image = Image.open(os.path.join(args.fake_path, file_name[0])).convert("RGB")
        
        fake_image = transform(fake_image).to("cuda")
        real_image = real_image.to("cuda")

        ssim_score = ssim(real_image, fake_image.unsqueeze(0), data_range=1).item()
        ssim_scores.append(ssim_score)

        if i % 10 == 0:
            print(f"(SSIM) Processed {i} images, time elapsed: {datetime.now() - start_time}, time remaining: {(datetime.now() - start_time) / (i + 1) * (len(fake_image_paths) - i - 1)}")

    print(f"Mean SSIM: {np.mean(ssim_scores):.6f}, std: {np.std(ssim_scores):.6f}")

    # results = pd.read_csv("results.csv")
    # # exp = args.fake_path.split("/")[-2]
    # exp = args.fake_path
    # if exp not in results["exp"].values:
    #     results = results._append({"exp": exp, f"SSIM-{args.coco_type}": np.mean(ssim_scores)}, ignore_index=True)
    # else:
    #     results.loc[results["exp"] == exp, f"SSIM-{args.coco_type}"] = np.mean(ssim_scores)
    # results = results.sort_values(by="exp")
    # results.to_csv("results.csv", index=False, columns=["exp", "LPIPS-5k", "MSE-5k", "PSNR-5k", "SSIM-5k", "CLIP-5k", "CLIP-D"])
if __name__ == "__main__":
    args = parse_args()
    main(args)