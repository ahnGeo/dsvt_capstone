# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm

from datasets import UCF101, HMDB51, Kinetics, KTH, Diving48
from models import get_vit_base_patch16_224, SwinTransformer3D, get_vmae_vit_base_patch16_224, get_gap_vit_base_patch16_224
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
# from torchsummary import summary as summary
import numpy as np

#^ for load ca_head
# class MultiCropWrapper(nn.Module):
#     def __init__(self, backbone, ca_head):
#         super(MultiCropWrapper, self).__init__()
#         # disable layers dedicated to ImageNet labels classification
#         backbone.fc, backbone.head = nn.Identity(), nn.Identity()
#         self.backbone = backbone
#         self.ca_head = ca_head
#         self.head = nn.Linear(768, 101)
        

#     def forward(self, x, head_only=False, loc=True):
#         output = self.backbone(x)
#         # logits = self.ca_head(output, loc)  #^ logits = self.norm(torch.cat([glo_tokens, loc_tokens], dim=1))
#         logits = self.ca_head(output, loc=False) 
        
#         return self.head(logits[:,:1]).reshape((-1, 101))


def eval_linear(args):

    config = load_config(args)
    
    # torch.manual_seed(config.RNG_SEED)
    # np.random.seed(config.RNG_SEED)
    utils.fix_random_seeds(config.RNG_SEED)
    
    utils.init_distributed_mode(args)
    
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # ============ preparing data ... ============
    # config.DATA.PATH_TO_DATA_DIR = f"{os.path.expanduser('~')}/repo/mmaction2/data/{args.dataset}/splits"
    # config.DATA.PATH_PREFIX = f"{os.path.expanduser('~')}/repo/mmaction2/data/{args.dataset}/videos"
    multiview_clips = config.TEST.NUM_ENSEMBLE_VIEWS
    multiview_crops = config.TEST.NUM_SPATIAL_CROPS
    
    config.TEST.NUM_SPATIAL_CROPS = 1
    if args.dataset == "ucf101":
        dataset_train = UCF101(cfg=config, mode="train", num_retries=10)
        dataset_val = UCF101(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_ENSEMBLE_VIEWS = multiview_clips
        config.TEST.NUM_SPATIAL_CROPS = multiview_crops
        multi_crop_val = UCF101(cfg=config, mode="val", num_retries=10)
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
        multi_crop_val = HMDB51(cfg=config, mode="val", num_retries=10)
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="test", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
        multi_crop_val = Kinetics(cfg=config, mode="val", num_retries=10)
    elif args.dataset == "kth":
        dataset_train = KTH(cfg=config, mode="train", num_retries=10)
        dataset_val = KTH(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
        multi_crop_val = KTH(cfg=config, mode="val", num_retries=10)
    elif args.dataset == "diving48" :
        dataset_train = Diving48(cfg=config, mode="train", num_retries=10)
        dataset_val = Diving48(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_ENSEMBLE_VIEWS = multiview_clips
        config.TEST.NUM_SPATIAL_CROPS = multiview_crops
        multi_crop_val=Diving48(cfg=config, mode="val", num_retries=10)
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(    #* shuffle=False
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    multi_crop_val_loader = torch.utils.data.DataLoader(     #* shuffle=False
        multi_crop_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    #!!!!!!!
    #! We replace the projection head over the SVT with a randomly initialized linear layer, initialize the SVT backbone

    # ============ building network ... ============
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=False, pretrained=args.img_pretrained)     #^ head = linear classifier
            model_embed_dim = model.embed_dim
        elif args.arch == "vmae_vit_base" :
            model = get_vmae_vit_base_patch16_224(cfg=config, no_head=False, init_scale=1)
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        elif args.arch == "gap_vit_base" :
            model = get_gap_vit_base_patch16_224(cfg=config, no_head=False)
        # elif args.arch == "dsvt_ca_head" :
        #     model = get_vit_base_patch16_224(cfg=config, no_head=True)     #^ head = linear classifier
        #     embed_dim = model.embed_dim
        #     print("embed_dim", embed_dim)
        #     model = MultiCropWrapper(model, 
        #             SelfPatchHead(embed_dim, model.num_heads, args.k_num, attention_type=args.patchHead_attention, sampling=args.patch_sampling))
        #             # SelfPatchHead(embed_dim, model.num_heads, 8, attention_type='divided_space_time',sampling='cubic'))
        else:
            raise Exception(f"invalid model: {args.arch}")
    
    cur_device = torch.cuda.current_device()
    model.cuda()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.dsvt or "vmae" in args.arch :
        model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
            )
    else :
        model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=False
            )
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model_without_ddp = model.module
    
    if "our" in args.pretrained_weights :
        ckpt = torch.load(args.pretrained_weights, map_location=torch.device('cuda'))
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]

        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.") and "head" not in x}

        #^ CLS not use, random init
        # if args.dsvt :
        #     renamed_checkpoint['cls_token'] = ckpt['ca_head.cls_token']
        #     renamed_checkpoint['norm.weight'] = ckpt['ca_head.norm.weight']
        #     renamed_checkpoint['norm.bias'] = ckpt['ca_head.norm.bias']
        # if load ca_head, no need set renamed_checkpoint, and msg = p_head, c_head
        
        msg = model_without_ddp.load_state_dict(renamed_checkpoint, strict=False)
        print("load pretrained model on eval_finetune.py")
        print(f"Loaded model with msg: {msg}")
        
    elif "pth" in args.pretrained_weights or "pyth" in args.pretrained_weights :  #* not any args, initialize with dino weights
        ckpt = torch.load(args.pretrained_weights, map_location=torch.device('cuda'))
        
        if "vitb" in args.pretrained_weights :   #* svt k400 pretrained weights from repo
            renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
            
        elif "TimeSformer" in args.pretrained_weights :
            ckpt = ckpt['model_state']
            renamed_checkpoint = {x[len("model."):]: y for x, y in ckpt.items() if x.startswith("model.") and not "head" in x}
            
        elif "vmae" in args.pretrained_weights :
            ckpt = ckpt['model']
            renamed_checkpoint = {x[len("encoder."):]: y for x, y in ckpt.items() if x.startswith("encoder.") and not "head" in x}
            renamed_checkpoint['fc_norm.weight'] = renamed_checkpoint['norm.weight']
            renamed_checkpoint['fc_norm.bias'] = renamed_checkpoint['norm.bias']
            
        else :
            renamed_checkpoint = ckpt

        msg = model_without_ddp.load_state_dict(renamed_checkpoint, strict=False)
        
        print("load pretrained model on eval_finetune.py")
        print(f"Loaded model with msg: {msg}")


    n = args.n_last_blocks

    if n == -1 : 
        print("#################################################\nlinear probing\n#################################################")
    elif n == 0 :
        print("#################################################\nfull finetuning\n#################################################")
    else :
        print(f"#################################################\nfinetuning with {n} layers\n#################################################")
    
    if n >= 1 :  #* finetune only last n blocks
        assert args.arch == "vit_base"
        # then
        on_block = [str(12 - i) for i in range(1, n + 1)]    #* finetune only the last layer -> on_block = ["11"]
        for name, param in model_without_ddp.named_parameters() :
            if "." not in name :
                param.requires_grad = False
            elif name.split(".")[1] in on_block : 
                param.requires_grad = True
            elif name == "norm.weight" or name == "norm.bias" or "head" in name :
                param.requires_grad = True
            elif "vmae" in args.arch :
                if "fc_norm" in name or "head" in name :
                    param.requires_grad = True
            else :
                param.requires_grad = False
    
    if n == -1 :  #* linear probe
        for name, param in model_without_ddp.named_parameters() :
            if name == 'head.weight' or name == 'head.bias' :
                param.requires_grad = True
            elif "gap" in args.arch and (name == 'norm.weight' or name == 'norm.bias') : #^ if gap is set, norm should be requires_grad True
                param.requires_grad = True 
            else :
                param.requires_grad = False

    for n, p in model.named_parameters() :
        print(n, p.requires_grad)

    params_groups = utils.get_params_groups(model)
    
    # set optimizer
    if not args.eval_linear :
        if "vmae" not in args.arch :
            optimizer = torch.optim.SGD(
                params_groups,
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
                momentum=0.9,
                weight_decay=0.0001 
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[11,14], gamma=0.1)
        else :
            print("set optimizer and scheduler like vmae setting")
            optimizer = torch.optim.AdamW(
                params_groups,
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
                weight_decay=0.05
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
            
    else :
        optimizer = torch.optim.SGD(
            params_groups,
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
        
    to_restore = {"epoch": 0, "best_acc": 0.}
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, optimizer, train_loader, epoch, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # val_loader.sampler.set_epoch(epoch)
            test_stats = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
            

    test_stats = validate_network_multi_view(multi_crop_val_loader, model, args.n_last_blocks,
                                             args.avgpool_patchtokens, config)
    print("multi view results : ")
    print(test_stats)

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.2f}".format(acc=best_acc))

    return best_acc

def train(model, optimizer, loader, epoch, avgpool):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    

    for (inp, target, sample_idx, meta) in metric_logger.log_every(loader, 20, header):
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)
            
        # output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, n, avgpool):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for (inp, target, sample_idx, meta) in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            # intermediate_output = model.get_intermediate_layers(inp, n)
            # output = [x[:, 0] for x in intermediate_output]
            # if avgpool:
            #     output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            # output = torch.cat(output, dim=-1)
            output = model(inp)
        loss = nn.CrossEntropyLoss()(output, target)

        if args.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if args.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if args.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network_multi_view(val_loader, model, n, avgpool, cfg):
    model.eval()
    test_meter = TestMeter(
        len(val_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        args.num_labels,
        len(val_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
        )
    test_meter.iter_tic()


    for cur_iter, (inp, target, sample_idx, meta) in tqdm(enumerate(val_loader), total=len(val_loader)):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        test_meter.data_toc()

        # forward
        with torch.no_grad():
            output = model(inp)
            

        output = output.cpu()
        target = target.cpu()
        sample_idx = sample_idx.cpu()

        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            output.detach(), target.detach(), sample_idx.detach()
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    test_meter.finalize_metrics()
    return test_meter.stats


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help=">=1 is turn on some layers for finetuning, 0 is full finetuning, -1 is linear probing")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_base', 'gap_vit_base', 'vame_vit_base', 'dsvt_ca_head'],
                        help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="ucf101", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")
    
    
    parser.add_argument('--img_pretrained', default=False, type=utils.bool_flag, help='load weights pretrained on an image model')
    parser.add_argument('--dsvt', default=False, type=utils.bool_flag, help='set find_unused_parameter True')
    parser.add_argument("--eval_linear", default=False, type=utils.bool_flag, help='set optimizer and scheduler for linear probing')
    

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    import time
    
    start = time.time()
    
    best_acc = eval_linear(args)

    end = time.time()
    
    import datetime
    import json
    import requests
    import os
    
    def notice_message(token, channel, text, attachments):
        attachments = json.dumps(attachments) 
        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+token},
            data={"channel": channel, "text": text ,"attachments": attachments})
    
    Token = 'xoxb-4894844378726-4901340369155-XVhgVGJvW9fhGQ4C2AVFKwK6' # 자신의 Token 입력
    job_name=os.environ["SLURM_JOB_NAME"] #자동으로 JOB name받아옴.
    cluster=os.environ["SLURM_SUBMIT_HOST"] #노드이름도 받아와줌.
    job_time= str(datetime.timedelta(seconds=(end-start))).split(".")[0]  #total_time_str#이건 내가 추가한변수
    attach_dict = {
    'color' : '#ff0000',
    'author_name' : 'Job Finish',
    'title' : job_name,
    'text' : cluster,
    }
    attach_list=[attach_dict] 
    contents=f"Training time is {job_time}\nTop 1 Accuracy is {best_acc}"
    notice_message(Token, "#notice", contents, attach_list)  # #notice = channel name
