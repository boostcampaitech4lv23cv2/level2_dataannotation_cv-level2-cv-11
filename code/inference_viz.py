import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect
from deteval import calc_deteval_metrics

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default='/opt/ml/input/data/ICDAR17_Korean')
    parser.add_argument('--json_path', default='/opt/ml/input/data/ICDAR17_Korean/ufo/train.json')
    parser.add_argument('--model_path', default= '/opt/ml/code/trained_models')
    parser.add_argument('--output_dir', default= '/opt/ml/code/predictions')

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, json_path ,input_size, batch_size):
    model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    with open(json_path) as f: data = json.load(f)
    for key in tqdm(data['images'].keys()):
        image_fpath  = osp.join(data_dir,'images', key)
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    with open(json_path) as f:  ufo_result = json.load(f)
    pred_sample_bboxes = {}
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist(), transcription="", language = "",
                                illegibility = False, orientation ="", word_tags = "") for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname]['words'] = dict(words_info)
        pred_sample_bboxes[image_fname] = bboxes
        

    return ufo_result, pred_sample_bboxes

def get_gt_bboxes__transcription(json_path):
    
    gt_sample_bboxes = {}
    transcription = {}
    
    with open(json_path) as f: data = json.load(f)
    for filename, tmp_dict in data['images'].items():
        idx = [ x for x in tmp_dict['words']]
        gt_sample_bboxes[filename] = [tmp_dict['words'][x]["points"] for x in idx]
        transcription[filename] = [tmp_dict['words'][x]["transcription"] for x in idx]

    return gt_sample_bboxes, transcription



def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = args.model_path

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    result, pred_bboxes_dict  = do_inference(model, ckpt_fpath, args.data_dir, args.json_path, args.input_size,
                                    args.batch_size)
    ufo_result['images'].update(result['images'])

    output_fname = 'output.json'
    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)
     
    gt_bboxes_dict, transcription = get_gt_bboxes__transcription(args.json_path)
     
    resDict = calc_deteval_metrics(pred_bboxes_dict,  gt_bboxes_dict, transcriptions_dict=transcription,
                         eval_hparams=None, bbox_format='rect', verbose=False)
    resDict = resDict['total']
    
    output_txtname = 'result.txt'
    with open(osp.join(args.output_dir, output_txtname), 'w') as f:
        f.write("#Result\n #Data: {}\n #model_pth: {}\n".format(args.data_dir, ckpt_fpath))
        f.write('DetEval/Precision: {:.6f}, DetEval/Recall: {:.6f}, DetEval/F1: {:.6f}\n'.format(
            resDict['precision'], resDict['recall'], resDict['hmean']))
    

if __name__ == '__main__':
    args = parse_args()
    main(args)