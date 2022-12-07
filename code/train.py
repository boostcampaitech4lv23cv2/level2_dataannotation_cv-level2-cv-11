import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

from reproducibility import fix_seed, seed_worker

import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    
    # seed
    parser.add_argument('--seed', type=int, default= 42, help="set random seed")
    
    # wandb args
    parser.add_argument('--name', type=str, help="wandb 실험 이름")
    parser.add_argument('--tags', default= None, nargs='+', type=str, help = "wandb 실험 태그")

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, name, tags, seed):
    
    train_dataset = SceneTextDataset(data_dir, split='random_split_ufo/train', image_size=image_size, crop_size=input_size)
    train_dataset = EASTDataset(train_dataset)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    # generator 재현성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator= seed_worker(seed))
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train = loss.item()
                epoch_loss += loss_train

                pbar.update(1)
                train_dict = {
                    'Train/Cls loss': extra_info['cls_loss'],
                    'Train/Angle loss': extra_info['angle_loss'],
                    'Train/IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(train_dict)
                
                # wandb: loss for step
                wandb.log(train_dict)
                     
        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))
        
        # wandb: loss for epoch
        wandb.log({"Train/Mean loss": epoch_loss / num_batches})
        wandb.log({"Train/Epoch":epoch})
        
        # 모델 평가
        print('\nModel Eval/Epoch {}:'.format(epoch + 1))
        model.eval()
        val_loss = {'cls_loss' : 0, 'angle_loss': 0, 'iou_loss': 0}
        val_dataset = SceneTextDataset(data_dir, split='random_split_ufo/val', image_size=image_size, crop_size=input_size)
        val_dataset = EASTDataset(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator= seed_worker(seed))
        val_num_batches = math.ceil(len(val_dataset) / batch_size)
        for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
            pbar.set_description('[Epoch {}]'.format(epoch + 1))

            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            
            for key in val_loss.keys():
                val_loss[key] += extra_info[key]
            
            
            # wandb: image gt and pred for val every epoch
            img_log = wandb.Image(img)
            gt_score_map_log = wandb.Image(gt_score_map)
            pred_score_map_log = wandb.Image(extra_info['score_map'])
            
            wandb.log({
                "Visual/img":img_log,
                "Visual/gt_score_map": gt_score_map_log,
                "Visual/pred_score_map_log": pred_score_map_log
                })
    
        val_dict = {
                'Val/Cls loss': val_loss['cls_loss']/val_num_batches,
                'Val/Angle loss':val_loss['angle_loss']/val_num_batches,
                'Val/IoU loss': val_loss['iou_loss']/val_num_batches
            }
        val_dict['Val/Mean loss'] =  val_dict['Val/Cls loss'] + val_dict['Val/Angle loss'] + val_dict['Val/IoU loss']
        
        print('Val/Mean loss: {:.4f}, Val/Cls loss: {:.4f}, Val/Angle loss: {:.4f}, Val/IoU loss: {:.4f}\n'.format(
            val_dict['Val/Mean loss'], val_dict['Val/Cls loss'], val_dict['Val/Angle loss'], val_dict['Val/IoU loss']))
         
        # wandb: loss for val every epoch
        wandb.log(val_dict)
        

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    assert args.name != None, "Error: 실험 이름을 적어주세요"
    assert args.tags != None, "Error: 실험 태그를 적어주세요"
    wandb.init(project="dataannotation", entity="miho", name=args.name, tags=args.tags)
    wandb.config.update(args)
    fix_seed(args.seed)
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
