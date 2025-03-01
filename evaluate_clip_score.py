import clip
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
        if not os.path.exists(coco_path):
            coco_path = "/data/datasets/coco/val2014"
        self.filenames = []
        self.prompts = []
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
    parser.add_argument("--coco_type", choices=["5k", "30k"])

    args = parser.parse_args()
    return args

def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    clip_fn, preprocess = clip.load("ViT-B/32")
    clip_fn = clip_fn.to("cuda")
    clip_fn.eval()

    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    
    dataset = COCODataset(csv_file=f"coco_val2014_{args.coco_type}.csv", coco_path="/local_datasets/coco/val2014", transform=transform)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    start_time = datetime.now()
    clip_scores = []
    for i, (_, prompt, filename) in enumerate(dataloader):
        fake_image = Image.open(os.path.join(args.fake_path, filename[0])).convert("RGB")
        fake_image = transform(fake_image).unsqueeze(0).to("cuda")

        image_embed = clip_fn.encode_image(fake_image)
        text_embed = clip_fn.encode_text(clip.tokenize(prompt).to("cuda"))

        clip_score = F.cosine_similarity(image_embed, text_embed).item()

        clip_scores.append(clip_score)

        if i % 100 == 0:
            print(f"(CLIP Score) Processed {i} images, time elapsed: {datetime.now() - start_time}, time remaining: {(datetime.now() - start_time) / (i + 1) * (len(dataloader) - i - 1)}")

    print(f"Mean CLIP score: {np.mean(clip_scores):.6f}, std: {np.std(clip_scores):.6f}")
    
    results = pd.read_csv("results.csv")
    exp = args.fake_path.split("/")[-2]
    if exp not in results["exp"].values:
        results = results._append({"exp": exp, f"CLIP-{args.coco_type}": np.mean(clip_scores)}, ignore_index=True)
    else:
        results.loc[results["exp"] == exp, f"CLIP-{args.coco_type}"] = np.mean(clip_scores)

    results.to_csv("results.csv", index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)