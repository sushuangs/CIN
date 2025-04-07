import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from thop import profile
from PIL import Image
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

import utils.utils as utils
from utils.yml import parse_yml, dict_to_nonedict, set_random_seed
from utils.yml import dict2str
from models.Network import Network


class ImageProcessingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = []
        self.rel_dirs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        for root, _, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
                    self.rel_dirs.append(rel_dir)

    def __len__(self):
        return len(self.image_paths)
    
    def generate_binary_seed(self, seed_str: str) -> int:
        seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def generate_binary_data(self, seed: int, length: int = 30) -> list:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, size=length).tolist()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            seed = self.generate_binary_seed(img_path)
            binary_data = self.generate_binary_data(seed, 30)
            return self.transform(img), idx, torch.tensor(binary_data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, idx

def batch_process(model, dataloader, output_root, device):

    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    ])

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            inputs, indices, message = batch
            
            valid_inputs = inputs.to(device)
            valid_indices = indices
            
            if valid_inputs.shape[0] == 0:
                continue

            normalized, message = valid_inputs, message.to(device)
            
            # outputs = model(normalized, message)
            outputs = model.encoder(normalized, message)
            
            denorm_outputs = denormalize(outputs.cpu().clamp(-1, 1))
            
            for tensor, idx in zip(denorm_outputs, valid_indices):
                orig_path = dataset.image_paths[idx]
                rel_dir = dataset.rel_dirs[idx]
                filename = os.path.basename(orig_path)
                
                output_dir = os.path.join(output_root, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                img = transforms.ToPILImage()(tensor)
                img.save(os.path.join(output_dir, filename))
            
            pbar.update(1)

if __name__ == "__main__":
    input_root = "/data/experiment/data/gtos128_all/val" # val
    output_root = "./CIN_35_gtos/val" # val
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    name = str("CIN")

    yml_path = '../codes/options/opt.yml'
    option_yml = parse_yml(yml_path)
    opt = dict_to_nonedict(option_yml)

    opt['path']['folder_temp'] = './temp'

    time_now_NewExperiment = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
    if opt['subfolder'] != None:
        subfolder_name = opt['subfolder'] + '/-'
    else:
        subfolder_name = ''
    #
    folder_str = opt['path']['logs_folder'] + name + '/' + subfolder_name + str(time_now_NewExperiment) + '-' + opt['train/test']
    log_folder = folder_str + '/logs'
    img_w_folder_tra = folder_str  + '/img/train'
    img_w_folder_val = folder_str  + '/img/val'
    img_w_folder_test = folder_str + '/img/test'
    loss_w_folder = folder_str  + '/loss'
    path_checkpoint = folder_str  + '/path_checkpoint'
    opt_folder = folder_str  + '/opt'
    opt['path']['folder_temp'] = folder_str  + '/temp'
    #
    path_in = {'log_folder':log_folder, 'img_w_folder_tra':img_w_folder_tra, \
                    'img_w_folder_val':img_w_folder_val,'img_w_folder_test':img_w_folder_test,\
                        'loss_w_folder':loss_w_folder, 'path_checkpoint':path_checkpoint, \
                            'opt_folder':opt_folder, 'time_now_NewExperiment':time_now_NewExperiment}
    network = Network(opt, device, path_in)
    model = network.cinNet.module
    model.eval()

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    batch_process(model, dataloader, output_root, device)