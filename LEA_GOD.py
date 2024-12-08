import os, random, argparse, tqdm, datetime, io
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from einops import rearrange, repeat
from sklearn.linear_model import Ridge
from scipy.stats import kendalltau, pearsonr
from PIL import Image

from src.AE_models import MaskGIT_VQModel
from src.transformer_models import *
from src.dataset import create_Kamitani_CLIP_dataset, CLIP_METRIC_dataset
from src.eval_metrics import get_eval_metric, get_eval_metric2
from utils.utils import Config, to_cuda, torch_init_model
from utils.logger import setup_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as torchvision_utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter



class Neural_Decoding_Template():
    def __init__(self, args, test_gpu=None):
        # --- init configurations ---
        config = Config(args.cfg_file)

        self.model_path = os.path.join(config.ROOT, 'checkpoints', config.NAME)
        self.check_file_exist(self.model_path)

        self.tblog_path = os.path.join(self.model_path, 'tblog')
        self.check_file_exist(self.tblog_path)

        config_path = os.path.join(self.model_path, args.cfg_file.split('/')[-1])
        if not os.path.exists(config_path):
            copyfile(args.cfg_file, config_path)

        if test_gpu is not None:
            config.GPU_ID = test_gpu
        self.use_cuda = config.GPU_ID is not None
        if self.use_cuda:
            # config.device = torch.device("cuda")
            config.device = torch.device('cuda:{}'.format(config.GPU_ID[0]))
        else:
            config.device = torch.device("cpu")
        config.cpu_device = torch.device("cpu")

        self.config = config
        self.set_random_seed(config.SEED)
        self.build_model()  # build model

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self,):
        raise NotImplementedError

    def build_model(self, ):
       raise NotImplementedError

    def build_optimizer(self, ):
        raise NotImplementedError

    @staticmethod
    def get_param_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {'Total_Param': total_num / 1e6, 'Trainable_Param': trainable_num / 1e6}

    def save_model(self, model, opt, path, epoch, iter, prefix=None):
        if prefix is not None:
            save_path = path + "_{}.pth".format(prefix)
        else:
            save_path = path + ".pth"

        print('\nsaving {}...\n'.format(save_path))
        all_saving = {'model': model.state_dict(),
                      'opt': opt.state_dict(),
                      'cur_epoch': epoch,
                      'cur_iter': iter, }
        torch.save(all_saving, save_path)

    def save_images(self, epoch, path, name, a1, a2, a3, fake_img):
        n, c, h, w = fake_img.size()
        samples = torch.FloatTensor(4*n, c, h, w).zero_()
        for i in range(n):
            samples[4*i+0] = a1[i].data
            samples[4*i+1] = a2[i].data
            samples[4*i+2] = a3[i].data
            samples[4*i+3] = fake_img[i].data

        images = torchvision_utils.make_grid(samples, nrow=8, padding=30, normalize=True)
        self.summary.add_image('samples', images, epoch)
        file_name = os.path.join(path, name)
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        torchvision_utils.save_image(samples, '%s/epoch_%d.png' % (file_name, epoch), nrow=8, padding=30, normalize=True)

    @staticmethod
    def gen_plot(fmri):
        """Create a pyplot plot and save to buffer."""
        # fmri: [D]
        plt.figure()
        plt.plot(np.arange(fmri.shape[0]), fmri)
        # plt.ylim(-2, 2)
        # plt.title("test")
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)

        image = Image.open(buf)
        image = transforms.ToTensor()(image).unsqueeze(0) # [1, C, H, W]
        plt.close()

        return image

    @staticmethod
    def check_file_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)



# Encoder: fMRI -> latent -> fMRI
# Ridge reg.: latent -> img feat
# Decoder: img feat -> RGB img
class Neural_Encoding_Decoding(Neural_Decoding_Template):
    def __init__(self, args, test_gpu=None):
        # super(Neural_Encoding_Decoding, self).__init__(args, test_gpu)
        # --- init configurations ---
        config_encoder = Config(args.cfg_file[0])
        config = Config(args.cfg_file[1])

        self.model_path = os.path.join(config.ROOT, 'checkpoints', config.NAME)
        self.check_file_exist(self.model_path)

        self.tblog_path = os.path.join(self.model_path, 'tblog')
        self.check_file_exist(self.tblog_path)

        config_path = os.path.join(self.model_path, args.cfg_file[1].split('/')[-1])
        if not os.path.exists(config_path):
            copyfile(args.cfg_file[1], config_path)

        if test_gpu is not None:
            config.GPU_ID = test_gpu
        self.use_cuda = config.GPU_ID is not None
        if self.use_cuda:
            # config.device = torch.device("cuda")
            config.device = torch.device('cuda:{}'.format(config.GPU_ID[0]))
        else:
            config.device = torch.device("cpu")
        config.cpu_device = torch.device("cpu")

        self.config_encoder = config_encoder
        self.config = config
        self.set_random_seed(config.SEED)
        self.build_model()  # build model
       
    def infer(self, ckpt_encoder, ckpt_decoder, num_samples, batch_size, num_step=11, gamma_mode='cosine', seed=666):
        self.set_random_seed(seed)

        data_cfg = self.config_encoder.Data
        model_cfg = self.config.Model
        g_config = Config(self.config.VQGAN_CFG)

        if ckpt_encoder:
            if os.path.exists(ckpt_encoder):
                print ('Loading %s Encoder weights ...' % ckpt_encoder)
                if self.use_cuda:
                    pre_state_dict = torch.load(ckpt_encoder, map_location=self.config.device)
                else:
                    pre_state_dict = torch.load(ckpt_encoder)
                torch_init_model(self.model_encoder, pre_state_dict['model'])
            else:
                print(ckpt_encoder, 'not Found')
                raise FileNotFoundError

        if ckpt_decoder:
            if os.path.exists(ckpt_decoder):
                print ('Loading %s Transformer weights ...' % ckpt_decoder)
                if self.use_cuda:
                    pre_state_dict = torch.load(ckpt_decoder, map_location=self.config.device)
                else:
                    pre_state_dict = torch.load(ckpt_decoder)
                torch_init_model(self.model_decoder, pre_state_dict['model'])
            else:
                print(ckpt_decoder, 'not Found')
                raise FileNotFoundError

        # -- Load VQGAN Model --
        g_path = self.config.VQGAN_MODEL
        if os.path.exists(g_path):
            print('Loading %s VQGAN weights ...' % g_path)
            if self.use_cuda:
                pre_state_dict = torch.load(g_path, map_location=self.config.device)
            else:
                pre_state_dict = torch.load(g_path)
            # pre_state_dict = {k.replace('module.', ''): v for k, v in pre_state_dict['g_model'].items()}
            torch_init_model(self.model_decoder.module.vqgan, pre_state_dict)
        else:
            print(g_path, 'not Found')
            raise FileNotFoundError

        # dataloader
        train_dataset, test_dataset = create_Kamitani_CLIP_dataset(path=data_cfg['path'], patch_size=data_cfg['patch_size'], image_size=data_cfg['image_size'],
                image_norm=data_cfg['norm'], random_flip=data_cfg['flip'], drop_rate=data_cfg['fmri_drop_rate'], subjects=data_cfg['sub'], include_nonavg_test=False,
                clip_name=model_cfg['clip_name'], clip_ckpt=model_cfg['clip_ckpt'], clip_cache=model_cfg['clip_cache'])
        test_loader = DataLoader(dataset=test_dataset,
                batch_size=batch_size,
                num_workers=8,
                drop_last=False,
                shuffle=False)
        
        # Build Ridge Regression
        ridge_loader = DataLoader(dataset=train_dataset, batch_size=128,
                num_workers=8, drop_last=False, shuffle=False)
        X = torch.FloatTensor(0).to(self.config.device)
        y = torch.FloatTensor(0).to(self.config.device)
        y2 = torch.FloatTensor(0).to(self.config.device)
        with torch.no_grad():
            cycle = 1
            for _ in range(cycle):
                for items in tqdm.tqdm(ridge_loader):
                    items = to_cuda(items, self.config.device)

                    img_feat = self.CLIP.encode_image(items['image4clip'], norm=model_cfg['clip_norm'])
                    img_feat2 = self.CLIP.encode_image(items['image4clip'], norm=False)
                    fmri_feat = self.model_encoder.module.transformer.forward_encoder(items['fmri'])

                    X = torch.cat([X, fmri_feat])
                    y = torch.cat([y, img_feat])
                    y2 = torch.cat([y2, img_feat2])

        clf = Ridge(alpha=500, fit_intercept=False)
        clf.fit(X[:, 0].cpu(), y.cpu())
        clf2 = Ridge(alpha=500, fit_intercept=True)
        clf2.fit(y.cpu(), X[:, 0].cpu())
        print ('Finish Ridge Regression.')

        # Start testing
        ckpt_name = os.path.splitext(ckpt_encoder)[0].split('/')[-1]
        eval_path = os.path.splitext(ckpt_encoder)[0].rsplit('/', 1)[0]
        self.save_path = os.path.join(eval_path, 'results', ckpt_name + '_' + datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        self.check_file_exist(self.save_path)

        self.model_encoder.eval()
        self.model_decoder.eval()
        all_samples = []
        all_samples2 = []
        all_fmri = []
        with torch.no_grad():
            all_test_latents = torch.FloatTensor(0).to(self.config.device)
            all_test_latents2 = torch.FloatTensor(0).to(self.config.device)
            for idx, items in enumerate(test_loader):
                print ('%d / %d' % (idx, len(test_loader)))
                items = to_cuda(items, self.config.device)
              
                # fmri -> image
                fmri_feat = self.model_encoder.module.transformer.forward_encoder(items['fmri']) # [B, 1, C]
                latent = torch.from_numpy(clf.predict(fmri_feat[:, 0].cpu())).float().to(self.config.device) # [B, C]

                # image -> fmri
                im_feat = self.CLIP.encode_image(items['image4clip'], norm=model_cfg['clip_norm']) # [B, C]
                latent2 = torch.from_numpy(clf2.predict(im_feat.cpu())).float().to(self.config.device) # [B, C]
                
                all_test_latents = torch.cat([all_test_latents, latent], dim=0)
                all_test_latents2 = torch.cat([all_test_latents2, latent2], dim=0)
            
            for idx, items in enumerate(test_loader):
                print ('%d / %d' % (idx, len(test_loader)))
                items = to_cuda(items, self.config.device)
                mri = items['fmri'][:, None].repeat(1, num_samples, 1, 1).flatten(end_dim=1)
                im = items['image'][:, None].repeat(1, num_samples, 1, 1, 1).flatten(end_dim=1)

                # fmri -> image
                mri_feat = self.model_encoder.module.transformer.forward_encoder(mri) # [B, 1, C]
                latent = torch.from_numpy(clf.predict(mri_feat[:, 0].cpu())).float().to(self.config.device) # [B, C]
                
                latent = (latent - torch.mean(all_test_latents, dim=0, keepdim=True)) / torch.std(all_test_latents, dim=0, keepdim=True)
                latent = latent * torch.std(y, dim=0, keepdim=True) + torch.mean(y, dim=0, keepdim=True)

                fake_tgt_img = self.model_decoder.module.predict_eval(latent, im, T=num_step, mode=gamma_mode)

                if g_config.Model['with_tanh']:
                    vis_norm = True
                    vis_range = (-1, 1)
                elif g_config.Data['norm']:
                    fake_tgt_img = torch.clamp(fake_tgt_img, min=-1, max=1)
                    vis_norm = True
                    vis_range = (-1, 1)
                else:
                    fake_tgt_img = torch.clamp(fake_tgt_img, min=0, max=1)
                    vis_norm = False
                    vis_range = None

                _, c, h, w = fake_tgt_img.shape
                fake_tgt_img = fake_tgt_img.view(-1, num_samples, c, h, w)
                samples = torch.cat([items['image'][:, None], fake_tgt_img], dim=1) # [B, N+1, 3, H, W]
                all_samples.append(samples.detach().cpu())

                # image -> fmri
                im_feat = self.CLIP.encode_image(items['image4clip'], norm=model_cfg['clip_norm']) # [B, C]
                # im_feat = self.CLIP.encode_image(items['image4clip'], norm=False) # [B, C]
                latent = torch.from_numpy(clf2.predict(im_feat.cpu())).float().to(self.config.device) # [B, C]

                latent = (latent - torch.mean(all_test_latents2, dim=0, keepdim=True)) / torch.std(all_test_latents2, dim=0, keepdim=True)
                latent = latent * torch.std(X[:, 0], dim=0, keepdim=True) + torch.mean(X[:, 0], dim=0, keepdim=True)

                fake_fmri = self.model_encoder.module.transformer.forward_decoder(latent[:, None]) # [B, L, 1]
                fake_fmri = fake_fmri.view(fake_fmri.shape[0], 1, -1)

                samples = torch.cat([items['fmri'], fake_fmri], dim=1) # [B, 2, L]
                all_fmri.append(samples)
                for idx in range(samples.shape[0]):
                    sub_samples = []
                    for idj in range(samples.shape[1]):
                        fig = self.gen_plot(samples[idx, idj].cpu().numpy()) # [1, C, H, W]
                        sub_samples.append(fig)
                    all_samples2.append(torch.cat(sub_samples, dim=0).detach().cpu()) # [2, C, H, W]

            del self.model_encoder
            del self.model_decoder

            grid_samples = torch.cat(all_samples, dim=0)
            grid_samples = rearrange(grid_samples, 'n b c h w -> (n b) c h w')
            torchvision_utils.save_image(grid_samples, '%s/sample_image.png' % (self.save_path), nrow=num_samples+1, padding=10, normalize=vis_norm, value_range=vis_range)

            grid_samples2 = torch.stack(all_samples2, dim=0)
            grid_samples2 = rearrange(grid_samples2, 'n b c h w -> (n b) c h w')
            torchvision_utils.save_image(grid_samples2, '%s/sample_fmri.png' % (self.save_path), nrow=2, padding=10, normalize=False)

    def build_model(self, ):
        self.model_encoder = Neural_fMRI2fMRI(self.config_encoder)
        self.model_decoder = Neural_Image2RGB_MaskGIT(self.config)
        self.CLIP = customized_CLIP(self.config.Model)

        if self.use_cuda:
            self.model_encoder.to(self.config.device)
            self.model_encoder = torch.nn.DataParallel(self.model_encoder, self.config.GPU_ID)
            self.model_encoder.eval()

            self.model_decoder.to(self.config.device)
            self.model_decoder = torch.nn.DataParallel(self.model_decoder, self.config.GPU_ID)
            self.model_decoder.eval()

            self.CLIP.to(self.config.device)
            self.CLIP.eval()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg_file', type=str, help='path to config file')
    args = parser.parse_args()

    # -* fmri -> latent -> ridge img feat -> RGB *-
    ckpt_encoder = 'checkpoints/GOD_sbj2/checkpoint.pth' # PATH_to_encoder_ckpt
    ckpt_decoder = 'checkpoints/ImageDecoder_MaskGIT/checkpoint.pth' # PATH_to_decoder_ckpt
    args.cfg_file = [
        'configs/fMRI_TransAE_GOD.yaml', # PATH_to_encoder_cfg
        'configs/MaskGIT_Transformer.yaml' # PATH_to_decoder_cfg
    ]
    engine = Neural_Encoding_Decoding(args, test_gpu=[0])
    engine.infer(ckpt_encoder=ckpt_encoder,
                ckpt_decoder=ckpt_decoder,
                num_samples=5,
                batch_size=4,
                num_step=11,
                gamma_mode='cosine',
                seed=666)
  

