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
import time
import shutil
import kornia
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
    ############
    #your model#
    #   start  #
    ############
import utils.utils as utils
from utils.yml import parse_yml, dict_to_nonedict, set_random_seed
from utils.yml import dict2str
from models.Network import Network

    ############
    #your model#
    #    end   #
    ############

def psnr_ssim_acc(image, H_img, L_img):
    # psnr
    H_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    L_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    # ssim
    H_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    L_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    return H_psnr, L_psnr, H_ssim, L_ssim

    ############
    #your model#
    #   start  #
    ############
def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
    ############
    #your model#
    #    end   #
    ############

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



if __name__ == "__main__":

    input_root = "/data/experiment/model/CIN/codes/CIN_35_gtos/val" # your path
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ############
    #your model#
    #   start  #
    ############
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

    ############
    #your model#
    #    end   #
    ############

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bitwise_avg_err_history = []
    with torch.no_grad():
        for data in dataloader:
            inputs, indices, message = data
            
            inputs = inputs.to(device)

            message = message.to(device)

            output_img = model.encoder(inputs, message)

            _, decoded_messages, _, _ = model.train_val_decoder(output_img, "")

            
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
            bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (
                    batch_size * 30)
            
            bitwise_avg_err_history.append(bitwise_avg_err)

    acc = 1 - np.mean(bitwise_avg_err_history)
    print('acc {:.4f}'.format(acc))