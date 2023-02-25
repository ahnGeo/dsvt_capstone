from sklearn.cluster import KMeans
import pickle
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torch import nn

from datasets.hmdb51 import HMDB51
from datasets.ucf101 import UCF101
from models import get_vit_base_patch16_224
from utils import utils
from utils.parser import load_config

import pickle


def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    config = load_config(args)
    # config.DATA.PATH_TO_DATA_DIR = f"{os.path.expanduser('~')}/repo/mmaction2/data/{args.dataset}/knn_splits"
    # config.DATA.PATH_PREFIX = f"{os.path.expanduser('~')}/repo/mmaction2/data/{args.dataset}/videos"
    
    #!!!!!!!!!!!! set crop one
    config.TEST.NUM_SPATIAL_CROPS = 1
    
    if args.dataset == "ucf101":
        # dataset_train = UCFReturnIndexDataset(cfg=config, mode="train", num_retries=10)
        dataset_val = UCFReturnIndexDataset(cfg=config, mode="val", num_retries=10)
    # elif args.dataset == "hmdb51":
        # dataset_train = HMDBReturnIndexDataset(cfg=config, mode="train", num_retries=10)
        # dataset_val = HMDBReturnIndexDataset(cfg=config, mode="val", num_retries=10)
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    # sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)  #* shuffle=False
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train,
    #     sampler=sampler,
    #     batch_size=args.batch_size_per_gpu,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")
    print(f"Data loaded with {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = get_vit_base_patch16_224(cfg=config, no_head=True)
    ckpt = torch.load(args.pretrained_weights)
    # select_ckpt = "teacher"
    renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
    msg = model.load_state_dict(renamed_checkpoint, strict=False)
    print(f"Loaded model with msg: {msg}")
    model.cuda()
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    # train_features = extract_features(model, data_loader_train)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val)

    if utils.get_rank() == 0:
        # train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # train_labels = torch.tensor([s for s in dataset_train._labels]).long()
    test_labels = torch.tensor([s for s in dataset_val._labels]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        
        print("Dump test features, shape : ", test_features.shape)
        print("Dump test labels, shape : ", test_labels.shape)
        print(test_labels)
        
        with open(os.path.join(args.dump_features, "testfeat.pkl"), 'wb') as f :
            pickle.dump(test_features.tolist(), f)
        with open(os.path.join(args.dump_features, "testlabels.pkl"), 'wb') as f :
            pickle.dump(test_labels.tolist(), f)
            
        # torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))            
        # torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        # torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        # torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    # return train_features, test_features, train_labels, test_labels
    return test_features, test_labels


@torch.no_grad()
def extract_features(model, data_loader):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            # if args.use_cuda:
            features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            # if args.use_cuda:
            features.index_copy_(0, index_all, torch.cat(output_l))
            # else:
            #     features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def kmeans_head(test_features, test_labels, k, num_classes=1000) :
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=101)
    kmeans.fit()
    pred = kmeans.predict(test_features)
    


class UCFReturnIndexDataset(UCF101):
    def __getitem__(self, idx):
        img, _, _, _ = super(UCFReturnIndexDataset, self).__getitem__(idx)
        return img, idx


class HMDBReturnIndexDataset(HMDB51):
    def __getitem__(self, idx):
        img, _, _, _ = super(HMDBReturnIndexDataset, self).__getitem__(idx)
        return img, idx
from torch.nn.functional import normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    with open(args.feature_path, 'rb') as f :
        data = pickle.load(f)    
    
    if type(data) == list :
        data = torch.tensor(data)
    
    data_norm = normalize(data, dim=1)
    
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(data)
    pred = kmeans.predict(data)
    pred = pred.tolist()
    pred = [str(pred[i]) for i in range(len(pred))]
    
    with open(args.output_path, 'w') as f :
        f.write('\n'.join(pred) + '\n')

    kmeans = KMeans(n_clusters=6, random_state=77)
    kmeans.fit(data_norm)
    pred = kmeans.predict(data_norm)
    pred = pred.tolist()
    pred = [str(pred[i]) for i in range(len(pred))]
    
    with open(args.output_path.split(".")[0] + "_norm.txt", 'w') as f :
        f.write('\n'.join(pred) + '\n')
