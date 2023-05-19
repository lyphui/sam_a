import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import datasets
import models
import utils
from statistics import mean
import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.distributed as dist

import torch.nn.functional as F
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    print('---------------make_data_loader',tag)
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr0(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'fmeasure1':
        metric_fn = utils.calc_fmeasure1
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'recall', 'precision'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4

def eval_iou(loader, model, eval_type=None):
    model.eval()

    if  eval_type == 'iou':  #TP, TN, FP, FN
        metric_fn = utils.calc_iou
        metric1, metric2, metric3, metric4 , metric5, metric6, metric7= 'intersection', 'union', 'new_target', 'TP', 'TN', 'FP', 'FN'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    intersection_meter = utils.Averager()
    union_meter = utils.Averager()
    target_meter = utils.Averager()
    TP_metric = utils.Averager()
    TN_metric = utils.Averager()
    FP_metric = utils.Averager()
    FN_metric = utils.Averager()


    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        with torch.no_grad():
            pred = torch.sigmoid(model.infer(inp))
            result1, result2, result3, result4, result5, result6,result7 = metric_fn(pred, batch['gt'])
#         print(result1, result2, result3, result4, result5, result6,result7,type(result4) 
        intersection_meter.add(result1)
        union_meter.add(result2)
        target_meter.add(result3)
        TP_metric.add(result4.item())  #intersection, union, new_target,TP, TN, FP, FN
        TN_metric.add(result5.item())
        FP_metric.add(result6.item())
        FN_metric.add(result7.item())

        if pbar is not None:
            pbar.update(1)
            
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    recall = TP_metric.sum / (TP_metric.sum + FN_metric.sum+ 1e-10)
    precision = TP_metric.sum / (TP_metric.sum + FP_metric.sum+ 1e-10)
    
    if pbar is not None:
        pbar.close()


    return  mIoU, mAcc, allAcc,recall,precision, 'mIoU', 'mAcc', 'allAcc','recall','precision'




def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 2
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    iii=0
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)
#         iii+=1
#         if iii>16:
#             break

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def load_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)

    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name]=param
            else:
                log(f"Warning: Shape mismatch for parameter '{name}'. Keeping original model parameter.")
        else:
            log(f"Ignored checkpoint parameter '{name}'")

    model.load_state_dict(model_state_dict)
    
def load_model2(model, checkpoint_path):

    state_dict = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name] = param
            else:
                log(f"Shapes mismatch for module '{name}'. Interpolating shapes...")
                log('model_state_dict[name].shape:' + str(model_state_dict[name].shape)+', param.shape:' + str(param.shape))
                if len(param.shape) == 2:
                    param=param.permute(1,0).unsqueeze(1).unsqueeze(1)
                    param = nn.functional.interpolate(param, size=model_state_dict[name].unsqueeze(0).shape[:2], mode='bilinear', align_corners=False)
                    model_state_dict[name] = param.squeeze().permute(1,0)
                else:
                    param=param.permute(0, 3, 1,2)
                    param = nn.functional.interpolate(param, size=model_state_dict[name].shape[1:-1], mode='bilinear', align_corners=False)
                    model_state_dict[name] = param.permute(0, 2, 3,1)
        else:
            log(f"Ignored checkpoint parameter '{name}'")

    model.load_state_dict(model_state_dict)

def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

#     pdb.set_trace()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))
#     print(model)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module
    
    assert config['load_mode'] in ['resize_load','skip_load','sfs'],f"{config['load_mode']} is not a valid load_mode"
    if config['load_mode']=='resize_load': 
        log(config['load_mode']+' from '+config['sam_checkpoint']+'...')
        load_model2(model, config['sam_checkpoint'])
    elif config['load_mode']=='skip_load': 
        log(config['load_mode']+' from '+config['sam_checkpoint']+'...')
        load_model(model, config['sam_checkpoint'])
    else:
        log('training from scratch...')
    
    
    log(config['finetune_mode']+' finetuning...')
    assert config['finetune_mode'] in ['fullfinetune','evp'],f"{config['finetune_mode']} is not a valid ft_mode"
    if config['finetune_mode']=='evp': 
        for name, para in model.named_parameters():
            if "image_encoder" in name and "prompt_generator" not in name:
                para.requires_grad_(False)
    
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log('model_grad_params:' + str(model_grad_params)+'; model_total_params:' + str(model_total_params))
        
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    
    if config.get('resume') is not None:
        resume_checkpoint = torch.load(config['resume'])
    #     print(resume_checkpoint.keys())
        model.load_state_dict(resume_checkpoint, strict=True)

    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if config.get('eval_type')=='iou': # mIoU, mAcc, allAcc,recall,precision, 'mIoU', 'mAcc', 'allAcc','recall','precision'
                result1, result2, result3, result4, result5, metric1, metric2, metric3, metric4, metric5 = eval_iou(val_loader, model, eval_type=config.get('eval_type'))
                if local_rank == 0:
                    log_info.append('val: {}={:.4f}'.format(metric1, result1))
                    writer.add_scalars(metric1, {'val': result1}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric2, result2))
                    writer.add_scalars(metric2, {'val': result2}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric3, result3))
                    writer.add_scalars(metric3, {'val': result3}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric4, result4))
                    writer.add_scalars(metric4, {'val': result4}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric5, result5))
                    writer.add_scalars(metric4, {'val': result5}, epoch)

                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')

                    t = timer.t()
                    prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                    t_epoch = utils.time_text(t - t_epoch_start)
                    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                    log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                    log(', '.join(log_info))
                    writer.flush()
            else:
                result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model, eval_type=config.get('eval_type'))
                if local_rank == 0:
                    log_info.append('val: {}={:.4f}'.format(metric1, result1))
                    writer.add_scalars(metric1, {'val': result1}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric2, result2))
                    writer.add_scalars(metric2, {'val': result2}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric3, result3))
                    writer.add_scalars(metric3, {'val': result3}, epoch)
                    log_info.append('val: {}={:.4f}'.format(metric4, result4))
                    writer.add_scalars(metric4, {'val': result4}, epoch)

                    if config['eval_type'] != 'ber':
                        if result1 > max_val_v:
                            max_val_v = result1
                            save(config, model, save_path, 'best')
                    else:
                        if result3 < max_val_v:
                            max_val_v = result3
                            save(config, model, save_path, 'best')

                    t = timer.t()
                    prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                    t_epoch = utils.time_text(t - t_epoch_start)
                    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                    log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                    log(', '.join(log_info))
                    writer.flush()
                    
                


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/glue-sam-vit-b.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    if config['load_mode']=='sfs':
        config['finetune_mode']='fullfinetune'
    save_path = os.path.join('./save', save_name, config['load_mode']+'_'+config['finetune_mode'])
    os.makedirs(save_path,exist_ok=True)
    main(config, save_path, args=args)
