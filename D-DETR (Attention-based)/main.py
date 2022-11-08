# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
from models.backbone import Backbone
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


# My:
#from utils_my import send_results
from torch import nn

from models import backbone
#from models import build_backbone # models/backbone 
#from models import Backbone

import math
#



def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')

    parser.add_argument('--lr_backbone', default=2e-5, type=float) # My: if >0 then backbone will be also trained
    #parser.add_argument('--lr_backbone', default=2e-4, type=float) 

    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    
    
    #parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--lr_linear_proj_mult', default=0.5, type=float)


    #parser.add_argument('--batch_size', default=2, type=int)
    #parser.add_argument('--batch_size', default=6, type=int) # 6 only works for some reason
    parser.add_argument('--batch_size', default=1, type=int)


    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    #parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--epochs', default=30, type=int) # For RN-50: 15 epochs ~ 12.5 hrs, 20 epochs ~ 16.7 hrs, 25 epochs ~ 20.9 hrs
                                                            # For RN-101: 1 epoch ~ 1 hr

    #parser.add_argument('--lr_drop', default=40, type=int) # drop every 40 epochs
    parser.add_argument('--lr_drop', default=7, type=int) # drop every N epochs

    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true') # My: 'store_true' sets defaut=False; if True = SGD, if False = AdamW (default)
    #parser.add_argument('--sgd', default=True, action='store_true')
    #parser.add_argument('--sgd', default=False, action='store_true') # same as the original line actually


    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # My:
    parser.add_argument('--pre_trained', default=True, action='store_true') # if a pre-trained model used
    #


    # * Backbone

    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    # parser.add_argument('--backbone', default='resnet101', type=str,
    #                     help="Name of the convolutional backbone to use") # Solution: first initialise with RN 50, and then change it to RN 101


    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels') # Feature layers from a CNN

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher

    # parser.add_argument('--set_cost_class', default=2, type=float,
    #                     help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=4, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)

    #parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=4, type=float)

    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    
    #parser.add_argument('--coco_path', default='./data/coco', type=str)
    #parser.add_argument('--coco_path', default='/home/dmitry.demidov/Documents/Datasets/MS COCO 2017/Original', type=str) # coco my
    parser.add_argument('--coco_path', default='/home/dmitry.demidov/Documents/Datasets/iSAID/iSAID_patches', type=str) # isaid

    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')

    #parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# My:
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
#

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
        
        # My:
        print("SGD optimiser is used")
        #

    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        
        # My:
        print("Adam optimiser is used")
        #

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_linear_proj_mult)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.5)

    if args.distributed:
        
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

            #My:
            # del checkpoint["model"]["class_embed.weight"]
            # del checkpoint["model"]["class_embed.bias"]
            # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            #


        # My:
        if args.pre_trained:
            #print(model_without_ddp)

            ## Change a number of output neurons (number of classes) 
                # to the number used for a pre-trained model (COCO = 91):

            # layers = model_without_ddp.class_embed
            # for i in range(6):
            #     layers[i] = nn.Linear(256, 91)
            # for i in range(6):
            #     model_without_ddp.class_embed[i] = nn.Linear(256, 91)
            for l in range(len(model_without_ddp.class_embed)):
                model_without_ddp.class_embed[l] = nn.Linear(args.hidden_dim, 91)
            

            ## Change backbone's layers used as transformer inputs 
                # to the layers used for a pre-trained model (l2, l3, l4, l4 with stride 2):
            
            # My: return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"} # My: originally commented out
            # Pre-trained: return_layers = {"layer2": "0", "layer3": "1", "layer4": "2", "layer4 + stride 2": "3"}

            strides = [8, 16, 32]
            #self.strides = [4, 8, 16, 32]

            num_channels = [512, 1024, 2048]
            #self.num_channels = [256, 512, 1024, 2048]


            if args.num_feature_levels > 1:
                num_backbone_outs = len(strides)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, args.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    ))
                for _ in range(args.num_feature_levels - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, args.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    ))
                    in_channels = args.hidden_dim
                model_without_ddp.input_proj = nn.ModuleList(input_proj_list)
            else:
                model_without_ddp.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(512, args.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    )])


            #print(model_without_ddp)
        #

        

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
 
        # My:
        if args.pre_trained:
            #print(model_without_ddp)

            ## Change back a number of output neurons (number of classes) 
                # to the number used for a pre-trained model (COCO = 91):

            # #for j in range(6):
            #     #model_without_ddp.class_embed[j] = nn.Linear(256, 16)
            # for l in enumerate(len(model_without_ddp.class_embed)):
            #     model_without_ddp.class_embed[l] = nn.Linear(256, 16)

            #     #model_without_ddp.class_embed.apply(init_weights)
            #     torch.nn.init.xavier_uniform_(model_without_ddp.class_embed[l].weight)
                
            #     #torch.nn.init.zeros_(model_without_ddp.class_embed[l].bias)
            #     num_classes = 16
            #     prior_prob = 0.01
            #     bias_value = -math.log((1 - prior_prob) / prior_prob)
            #     model_without_ddp.class_embed[l].bias.data = torch.ones(num_classes) * bias_value
            
            for l in range(len(model_without_ddp.class_embed)):
                model_without_ddp.class_embed[l] = nn.Linear(args.hidden_dim, 16)
            
            #model_without_ddp.class_embed.apply(init_weights)
            torch.nn.init.xavier_uniform_(model_without_ddp.class_embed[l].weight)
            
            #torch.nn.init.zeros_(model_without_ddp.class_embed[l].bias)
            num_classes = 16
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            model_without_ddp.class_embed[l].bias.data = torch.ones(num_classes) * bias_value


            ## Change back backbone's layers used as transformer inputs 
                # to the layers used for a pre-trained model (l2, l3, l4, l4 with stride 2):
                
            # My: return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"} # My: originally commented out
            # Pre-trained: return_layers = {"layer2": "0", "layer3": "1", "layer4": "2", "layer4 + stride 2": "3"}

            #strides = [8, 16, 32]
            strides = [4, 8, 16, 32]

            #num_channels = [512, 1024, 2048]
            num_channels = [256, 512, 1024, 2048]


            if args.num_feature_levels > 1:
                num_backbone_outs = len(strides)
                input_proj_list = []
                for _ in range(num_backbone_outs):
                    in_channels = num_channels[_]
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, args.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    ))
                for _ in range(args.num_feature_levels - num_backbone_outs):
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, args.hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    ))
                    in_channels = args.hidden_dim
                model_without_ddp.input_proj = nn.ModuleList(input_proj_list)
            else:
                model_without_ddp.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(num_channels[0], args.hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, args.hidden_dim),
                    )])

            for proj in model_without_ddp.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)


            #print(model_without_ddp)
            #


            '''
            args.backbone = 'resnet101'
            model_without_ddp.backbone = backbone.build_backbone(args)
            
            #model_without_ddp.backbone.Backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
            '''
            model.to(device)


            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                # import copy
                # p_groups = copy.deepcopy(optimizer.param_groups)
                # optimizer.load_state_dict(checkpoint['optimizer'])
                # for pg, pg_old in zip(optimizer.param_groups, p_groups):
                #     pg['lr'] = pg_old['lr']
                #     pg['initial_lr'] = pg_old['initial_lr']

                #print(optimizer.param_groups) # My: original line commented out

                #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                # My:
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.gamma = args.lr_linear_proj_mult
                #lr_scheduler.gamma = 0.5
                #
                
                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                
                #args.override_resumed_lr_drop = True
                args.override_resumed_lr_drop = False

                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                    
                    # My:
                    #lr_scheduler.gamma = args.lr_linear_proj_mult
                    #

                #lr_scheduler.step(lr_scheduler.last_epoch)
                if args.override_resumed_lr_drop:
                    lr_scheduler.step(lr_scheduler.last_epoch)

                #args.start_epoch = checkpoint['epoch'] + 1 # My: original line commented out
                if args.override_resumed_lr_drop: # better if not xxx ?
                    args.start_epoch = checkpoint['epoch'] + 1
            #
        


    
        else:
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                import copy
                p_groups = copy.deepcopy(optimizer.param_groups)
                optimizer.load_state_dict(checkpoint['optimizer'])
                for pg, pg_old in zip(optimizer.param_groups, p_groups):
                    pg['lr'] = pg_old['lr']
                    pg['initial_lr'] = pg_old['initial_lr']

                #print(optimizer.param_groups) # My: original line commented out

                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                
                # My:
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.gamma = args.lr_linear_proj_mult
                #lr_scheduler.gamma = 0.5
                #
                
                # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
                
                #args.override_resumed_lr_drop = True
                args.override_resumed_lr_drop = False

                if args.override_resumed_lr_drop:
                    print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                    lr_scheduler.step_size = args.lr_drop
                    lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
                    
                    # My:
                    #lr_scheduler.gamma = args.lr_linear_proj_mult
                    #

                #lr_scheduler.step(lr_scheduler.last_epoch)
                if args.override_resumed_lr_drop:
                    lr_scheduler.step(lr_scheduler.last_epoch)

                #args.start_epoch = checkpoint['epoch'] + 1 # My: original line commented out
                if args.override_resumed_lr_drop: # better if not xxx
                    args.start_epoch = checkpoint['epoch'] + 1


        #My:
        #print(model)
        output_dir = Path(args.output_dir)
        if args.output_dir: #and utils.is_main_process():
            with (output_dir / "model.txt").open("a") as f:
                f.write(str(model) + "\n")
        #


        # check the resumed model
        # if not args.eval:
        #     test_stats, coco_evaluator = evaluate(
        #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #     )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    print("Start training")

    # My:
    #send_results(text='Start')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
            # # extra checkpoint before LR drop and every 5 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
            #     checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

                # My:
                #send_results(text=json.dumps(log_stats))


            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']

                    #if epoch % 50 == 0:
                    if epoch % 10 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
