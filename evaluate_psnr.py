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
    
def calculate_psnr(real_image: torch.Tensor,
                   fake_image: torch.Tensor):
    mse = F.mse_loss(fake_image, real_image)
    if mse == 0:
        return 100
    else:
        return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

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

    start_time = datetime.now()
    psnr_scores = []
    for i, (real_image, _, filename) in enumerate(dataloader):
        real_image = real_image.to("cuda")
        fake_image = Image.open(os.path.join(args.fake_path, filename[0])).convert("RGB")
        fake_image = transform(fake_image).unsqueeze(0).to("cuda")

        psnr_score = calculate_psnr(real_image, fake_image)
        psnr_scores.append(psnr_score)

        if i % 10 == 0:
            print(f"(PSNR) Processed {i} images, time elapsed: {datetime.now() - start_time}, time remaining: {(datetime.now() - start_time) / (i + 1) * (len(dataloader) - i - 1)}")

    print(f"Mean PSNR: {np.mean(psnr_scores):.6f}, std: {np.std(psnr_scores):.6f}")

    # results = pd.read_csv("results.csv")
    # # exp = args.fake_path.split("/")[-2]
    # exp = args.fake_path
    # if exp not in results["exp"].values:
    #     results = results._append({"exp": exp, f"PSNR-{args.coco_type}": np.mean(psnr_scores)}, ignore_index=True)
    # else:
    #     results.loc[results["exp"] == exp, f"PSNR-{args.coco_type}"] = np.mean(psnr_scores)
    # results = results.sort_values(by="exp")
    # results.to_csv("results.csv", index=False, columns=["exp", "LPIPS-5k", "MSE-5k", "PSNR-5k", "SSIM-5k", "CLIP-5k", "CLIP-D"])
if __name__ == "__main__":
    args = parse_args()
    main(args)