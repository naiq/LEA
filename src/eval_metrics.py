from os import get_inheritable
import tqdm, math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import ViT_H_14_Weights, vit_h_14
import torch
from einops import rearrange
from torchmetrics.functional import accuracy
from PIL import Image
from scipy.stats import binom, pearsonr

def larger_the_better(gt, comp):
    return gt > comp

def smaller_the_better(gt, comp):
    return gt < comp


def clip_metric(img1, img2):
    # img1 (pred): n, 1, 1, c
    # img2 (gt): n, 1, 1, c
    img1 = img1[:, 0, 0]
    img2 = img2[:, 0, 0]

    r = np.corrcoef(img2, img1) #cosine_similarity(ground_truth, predictions)
    r = r[:len(img2), len(img2):]  # rows: groundtruth, columns: predicitons
    congruents = np.diag(r)

    # for each column (predicition) we should count the number of rows (groundtruth) that the value is lower than the congruent (e.g. success).
    success = r < congruents
    success_cnt = np.sum(success, 0)

    # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
    perf = np.mean(success_cnt) / (len(img2)-1)
    p = 1 - binom.cdf(perf*len(img2)*(len(img2)-1), len(img2)*(len(img2)-1), 0.5)
    
    return perf

def snr_metric(img1, img2):
    noise_signal = img1 - img2
    clean_signal = img2
    sum1 = np.sum(clean_signal ** 2)
    sum2 = np.sum(noise_signal ** 2)
    return 10 * math.log10(sum1 / sum2)
    # return 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))

def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()

def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)

def identity(x):
    return x

class psm_wrapper:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

    @torch.no_grad()
    def __call__(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = rearrange(img1, 'w h c -> c w h')
            img2 = rearrange(img2, 'w h c -> c w h')
        img1 = img1 / 127.5 - 1.0
        img2 = img2 / 127.5 - 1.0
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        return self.lpips(torch.FloatTensor(img1).to(self.device), torch.FloatTensor(img2).to(self.device)).item()

class fid_wrapper:
    def __init__(self, feature_dim, device='cuda'):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=feature_dim).to(self.device)

    @torch.no_grad()
    def __call__(self, pred_imgs, gt_imgs):
        assert pred_imgs.ndim == 4 and gt_imgs.ndim == 4
        self.fid.reset()

        pred = torch.tensor(pred_imgs, dtype=torch.uint8).permute(0, 3, 1, 2).to(self.device)
        gt = torch.tensor(gt_imgs, dtype=torch.uint8).permute(0, 3, 1, 2).to(self.device)

        self.fid.update(gt, real=True)
        self.fid.update(pred, real=False)

        return self.fid.compute().item()


def pair_wise_score(pred_imgs, gt_imgs, metric, is_sucess):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = [img for i, img in enumerate(gt_imgs) if i != idx]
        count = 0
        for comp in rest:
            comp_score = metric(pred, comp)
            if is_sucess(gt_score, comp_score):
                count += 1
        corrects.append(count / len(rest))
    return corrects

def pair_wise_score2(pred_imgs, gt_imgs, metric, is_sucess):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    # assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0 # !!!!!
    # assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    if isinstance(metric, fid_wrapper):
        fid_score = metric(pred_imgs, gt_imgs)
        corrects.append(fid_score)
    else:
        for idx, pred in enumerate(pred_imgs):
            gt = gt_imgs[idx]
            gt_score = metric(pred, gt)

            rest = [img for i, img in enumerate(gt_imgs) if i != idx]
            count = 0
            for comp in rest:
                comp_score = metric(pred, comp)
                if is_sucess(gt_score, comp_score):
                    count += 1
            corrects.append(count / len(rest))
    return corrects

def n_way_scores(pred_imgs, gt_imgs, metric, is_sucess, n=2, n_trials=100):
    # pred_imgs: n, w, h, 3
    # gt_imgs: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.
    assert len(pred_imgs) == len(gt_imgs)
    assert n <= len(pred_imgs) and n >= 2
    assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0
    assert isinstance(metric, fid_wrapper) == False, 'FID not supported'
    corrects = []
    for idx, pred in enumerate(pred_imgs):
        gt = gt_imgs[idx]
        gt_score = metric(pred, gt)
        rest = np.stack([img for i, img in enumerate(gt_imgs) if i != idx])
        correct_count = 0
        for _ in range(n_trials):
            n_imgs_idx = np.random.choice(len(rest), n-1, replace=False)
            n_imgs = rest[n_imgs_idx]
            count = 0
            for comp in n_imgs:
                comp_score = metric(pred, comp)
                if is_sucess(gt_score, comp_score):
                    count += 1
            if count == len(n_imgs):
                correct_count += 1
        corrects.append(correct_count / n_trials)
    return corrects

def metrics_only(pred_imgs, gt_imgs, metric, *args, **kwargs):
    # assert np.min(pred_imgs) >= 0 and np.min(gt_imgs) >= 0 # !!!!!

    return metric(pred_imgs, gt_imgs)

@torch.no_grad()
def n_way_top_k_acc(pred, class_id, n_way, num_trials=1000, top_k=1):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])

        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device),
                    task='multiclass', num_classes=n_way, top_k=top_k)
        acc_list.append(acc.item())

    return np.mean(acc_list), np.std(acc_list)

@torch.no_grad()
def get_n_way_top_k_acc(pred_imgs, ground_truth, n_way, num_trials, top_k, device, return_std=False):
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights)
    preprocess = weights.transforms()
    model = model.to(device)
    model = model.eval()

    acc_list = []
    std_list = []
    for pred, gt in zip(pred_imgs, ground_truth):
        pred = preprocess(Image.fromarray(pred.astype(np.uint8))).unsqueeze(0).to(device)
        gt = preprocess(Image.fromarray(gt.astype(np.uint8))).unsqueeze(0).to(device)
        gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()
        pred_out = model(pred).squeeze(0).softmax(0).detach()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        acc_list.append(acc)
        std_list.append(std)

    if return_std:
        return acc_list, std_list
    return acc_list

def get_similarity_metric(img1, img2, method='pair-wise', metric_name='mse', device='cuda', **kwargs):
    # img1: n, w, h, 3
    # img2: n, w, h, 3
    # all in pixel values: 0 ~ 255
    # return: list of scores 0 ~ 1.

    # if img1.shape[-1] != 3:
    #     img1 = rearrange(img1, 'n c w h -> n w h c')
    # if img2.shape[-1] != 3:
    #     img2 = rearrange(img2, 'n c w h -> n w h c')

    if method == 'pair-wise':
        eval_procedure_func = pair_wise_score2
    elif method == 'n-way':
        eval_procedure_func = n_way_scores
    elif method == 'metrics-only':
        eval_procedure_func = metrics_only
    elif method == 'class':
        return get_n_way_top_k_acc(img1, img2, device=device, **kwargs)
    else:
        raise NotImplementedError

    if metric_name == 'mse':
        metric_func = mse_metric
        decision_func = smaller_the_better
    elif metric_name == 'pcc':
        metric_func = pcc_metric
        decision_func = larger_the_better
    elif metric_name == 'ssim':
        metric_func = ssim_metric
        decision_func = larger_the_better
    elif metric_name == 'snr':
        metric_func = snr_metric
        decision_func = larger_the_better
    elif metric_name == 'clip':
        metric_func = clip_metric
        decision_func = larger_the_better
    elif metric_name == 'psm':
        metric_func = psm_wrapper(device=device)
        decision_func = smaller_the_better
    elif metric_name == 'fid':
        metric_func = fid_wrapper(feature_dim=64, device=device) # be one of [64, 192, 768, 2048]
        decision_func = smaller_the_better
    else:
        raise NotImplementedError

    return eval_procedure_func(img1, img2, metric_func, decision_func, **kwargs)

# for Mind-Vis
def get_eval_metric(samples, avg=True, metric_list=None, device='cuda'):
    if metric_list is None:
        metric_list = ['fid', 'mse', 'pcc', 'ssim', 'psm']
    assert all([m in ['mse', 'pcc', 'ssim', 'psm', 'fid'] for m in metric_list])
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            torch.cuda.empty_cache()
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m, device=device)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    for s in samples_to_run:
        torch.cuda.empty_cache()
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=1000, top_k=1, device=device)
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

# for our work
def get_eval_metric2(samples, avg=True, metric_list=None, metric_acc=True, device='cuda'):
    if metric_list is None:
        metric_list = ['fid', 'mse', 'pcc', 'ssim', 'psm', 'snr', 'clip']
    assert all([m in ['mse', 'pcc', 'ssim', 'psm', 'fid', 'snr', 'clip'] for m in metric_list])
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    print ('  --> pair-wise evaluation...')
    for m in tqdm.tqdm(metric_list):
        res_part = []
        for s in samples_to_run:
            torch.cuda.empty_cache()
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            if m in ['clip', 'ssim']:
                res = get_similarity_metric(pred_images, gt_images, method='metrics-only', metric_name=m, device=device)
            else:
                res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m, device=device)
            res_part.append(np.mean(res))
        res_list.append((np.mean(res_part),np.min(res_part),np.max(res_part)))
    
    if metric_acc:
        print ('  --> classification evaluation...')
        res_part = []
        for s in tqdm.tqdm(samples_to_run):
            torch.cuda.empty_cache()
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, 'class', None,
                            n_way=50, num_trials=1000, top_k=1, device=device)
            res_part.append(np.mean(res))
  
        res_list.append((np.mean(res_part), np.min(res_part), np.max(res_part)))
        metric_list.append('top-1-class')

    return res_list, metric_list


