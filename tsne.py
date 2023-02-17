from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--features_path', type=str)
parser.add_argument('--labels_path', type=str)
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--kmeans_results_path', type=str, default=None)
parser.add_argument('--display_cls_list', nargs='+', type=int, default=[])
parser.add_argument('--display_method', type=str, choices=['base', 'sub_in_all', 'sub', 'sub_marking_all_cls_idx', "sub_marking_all_vid_idx", 'detail_with_vid_idxs'])
parser.add_argument('--detail_cls_list', nargs='+', type=int, default=[], help="select classes desired to be marked vid idxs in 'detail_with_vid_idxs' ver")
parser.add_argument('--num_vids_per_cls', type=int, default=None)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_path_with_name', type=str, default=None)

def load_features_and_labels(features_path, labels_path) : 
    # load your dataset
    with open(features_path, 'rb') as f :
        features = pickle.load(f)
    print("data shape : ({}, {})".format(len(features), len(features[0])))

    with open(labels_path, 'r') as f :
        label_idxs = f.readlines()
        label_idxs = [int(line.split()[1]) for line in label_idxs]
    
    return features, label_idxs


def get_label_texts(dataset) :
    if dataset == 'ucf101' :
        with open("/data/ahngeo11/mmaction2/data/ucf101/annotations/classInd.txt", 'r') as f :
            class_bind = f.readlines()
    elif dataset == 'kth' :
        with open("/data/ahngeo11/svt/datasets/annotations/KTH-class-idx.txt", 'r') as f :
            class_bind = f.readlines()
            
    label_texts = dict()
    for line in class_bind :
        idx, label = int(line.split()[0]), line.split()[1]
        label_texts[idx] = label
    
    return label_texts  
    
        
def load_kmeans_results(results_path) :
    with open(results_path, 'r') as f :
        cluster_labels = f.readlines()
    
    for i in range(len(cluster_labels)) :
        cluster_labels[i] = int(cluster_labels[i].strip('\n'))
    
    return cluster_labels   #* list


def filtering_inputs(display_list, features, labels) :
    filtered_features, filtered_labels = [], []
    for i in range(len(labels)) :
        if labels[i] in display_list :
            filtered_features.append(features[i])
            filtered_labels.append(labels[i])
    
    filtered_features = np.array(filtered_features)
    return filtered_features, filtered_labels   #* np.array, list


def marking_cls_idx_one_sample_all_classes(plt, data, labels, n_classes) :
    #* marking one sample for all classes
    n_classes =  n_classes
    for i in range(n_classes):
        idx = labels.index(i)
        x = data[idx, 0]
        y = data[idx, 1]
        plt.text(x, y, str(i), fontsize=8)

def marking_cls_idx_one_sample_display_list(plt, data, labels, display_list) :
    #* marking one sample for each class in display_list
    #* display_list = list[int]
    for i in display_list :
        idx = labels.index(i)
        x = data[idx, 0]
        y = data[idx, 1]
        plt.text(x, y, str(i), fontsize=8)

def marking_cls_idx_every_sample_display_list(plt, data, labels, display_list) :
    #* marking every sample in display_list :
    for i in range(len(labels)) :
        if labels[i] in display_list :
            x = data[i, 0]
            y = data[i, 1]
            plt.text(x, y, labels[i], fontsize=7)

def marking_vid_idx_every_sample_display_list(plt, data, labels, display_list, display_version="next") :
    #* marking with video idx : base ver. 
    video_idx = 0
    
    for i in range(len(labels)) :
        if labels[i] in display_list :
            if i == labels.index(labels[i]) :   #* if i_th sample is the first sample of each class, set video_idx = 0
                video_idx = 0
            x = data[i, 0] 
            y = data[i, 1]
            
            if display_version == "top_bottom" :
                #* top - cls_idx, bottom - vid_idx
                plt.text(x, y, str(labels[i]), fontsize=6, verticalalignment="bottom", horizontalalignment="center")
                plt.text(x, y, str(video_idx), fontsize=5.5, verticalalignment="top", horizontalalignment="center")
            elif display_version == "next" :
                #* label like (,)
                plt.text(x, y, "({},{})".format(labels[i], video_idx), fontsize=6, verticalalignment="bottom", horizontalalignment="center")
            
            video_idx += 1    #* ann is written in order of vid_idxs

def finding_videos_in_display_list(plt, data, labels, display_list, display_vid_list) :
    #* detail ver.
    video_idx = 0
    plt_texts = []
    
    for i in range(len(labels)) :
        if labels[i] in display_list :
            if i == labels.index(labels[i]) :   #* if i is the start idx of that class in list 'label_idxs'
                video_idx = 0
            x = data[i, 0] 
            y = data[i, 1]
            
            #* label like (,)
            if video_idx in display_vid_list :  
                plt_texts.append(plt.text(x, y, "({},{})".format(labels[i], video_idx), fontsize=6, verticalalignment="bottom", horizontalalignment="center"))
            
            video_idx += 1
            
    return plt_texts


if __name__ == "__main__" :

    args = parser.parse_args()
    
    # ! python tsne.py --features_path /data/ahngeo11/svt/outputs/ucf101/simple/testfeat.pkl \
    # !     --labels_path /data/ahngeo11/svt/datasets/annotations/ucf101_val_split_1_videos_simple.txt \
    # !     --display_cls_list 28 35 39 50 62 67 78 90 --display_method detail_with_vid_idxs \
    # !     --detail_cls_list 39 --num_vids_per_cls 7 --output_dir /data/ahngeo11/svt/visualization/img/svt-k400-ucf101-noft 
    
    # python tsne.py --features_path /data/ahngeo11/svt/outputs/kth/cut-one/testfeat.pkl \
    #     --labels_path /data/ahngeo11/svt/datasets/annotations/kth_fe_videos.txt --display_method base \
    #     --num_vids_per_cls 100 --dataset kth \
    #     --output_path_with_name /data/ahngeo11/svt/visualization/img/svt-k400-kth-noft/cut-one-base.png 
    
    #? check dataset
    if args.dataset == "ucf101" :
        n_classes = 101
    elif args.dataset == "kth" :
        n_classes = 6
    
    #? load features
    data, labels = load_features_and_labels(args.features_path, args.labels_path)   #* data[i] = feature tensor of i_th sample, labels[i] = label of i_th sample
    
    if args.kmeans_results_path is not None :
        cluster_labels = load_kmeans_results(args.kmeans_results_path)   #* display clustering results in plt
        
    #? select classes to display
    if args.display_method == 'base' or args.display_method == 'sub_in_all' :
        background_display_cls_list = range(n_classes)
    else :
        background_display_cls_list = args.display_cls_list
    data, labels = filtering_inputs(background_display_cls_list, data, labels)
    
    #? perform t-SNE
    print("before tsne shape : ", data.shape)
    
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data)
    
    print("after tsne shape : ", data_tsne.shape)
    
    #? plot the results
    cmap = plt.get_cmap('Paired')

    if args.kmeans_results_path is not None : 
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=cluster_labels, cmap=cmap, s=10)
    elif args.display_method == 'base' or args.display_method == 'sub_in_all' :
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap=cmap, s=10)
    else :
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap=cmap, s=30, marker='*')
    
    #? marking texts on plt
    display_cls_list = args.display_cls_list
    
    if args.display_method == 'base' :
        marking_cls_idx_one_sample_all_classes(plt, data_tsne, labels, n_classes)
    elif args.display_method == 'sub_in_all' :
        marking_cls_idx_every_sample_display_list(plt, data_tsne, labels, display_cls_list)
    elif args.display_method == 'sub' :    
        marking_cls_idx_one_sample_display_list(plt, data_tsne, labels, display_cls_list)
    elif args.display_method == 'sub_marking_all_cls_idx' :
        marking_cls_idx_every_sample_display_list(plt, data_tsne, labels, display_cls_list)
    elif args.display_method == "sub_marking_all_vid_idx" :
        marking_vid_idx_every_sample_display_list(plt, data_tsne, labels, display_cls_list)
    
    #? title
    label_texts = get_label_texts(args.dataset)   #* label_text[0] = 'ApplyEyeMakeup'

    if args.display_method == "base" :  
        pass
    elif len(display_cls_list) < 7 :
        plt.title(' '.join([label_texts[label_idx]+"({})".format(label_idx) for label_idx in display_cls_list]), fontsize=7)
    else :
        #* divide title into two lines
        title = ' '.join([label_texts[label_idx]+"({})".format(label_idx) for label_idx in display_cls_list[:(len(display_cls_list)//2)]]) + '\n' + \
            ' '.join([label_texts[label_idx]+"({})".format(label_idx) for label_idx in display_cls_list[(len(display_cls_list)//2):]])
        plt.title(title, fontsize=7)

    #? save
    if args.output_path_with_name is not None :
        print("save ", args.output_path_with_name)
        plt.savefig(args.output_path_with_name)   
    elif args.display_method == "base" :
        print("save " + os.path.join(args.output_dir, "base.png"))  
        plt.savefig(os.path.join(args.output_dir, "base.png"))
    else :
        img_name = '&'.join([str(elem) for elem in display_cls_list])
        
        if not os.path.exists(os.path.join(args.output_dir, img_name)) :    #* like svt/visualization/img/7&68&89/img.png
            os.makedirs(os.path.join(args.output_dir, img_name))
        output_dir = os.path.join(args.output_dir, img_name)
        
        if args.display_method == "sub_in_all" :
            print("save " + os.path.join(output_dir, img_name + "_inall.png"))
            plt.savefig(os.path.join(output_dir, img_name + "_inall.png"))
        elif args.display_method == "sub" :
            print("save " + os.path.join(output_dir, img_name + ".png"))
            plt.savefig(os.path.join(output_dir, img_name + ".png"))
        elif args.display_method == "sub_marking_all_cls_idx" :
            print("save " + os.path.join(output_dir, img_name + "_markingall_cls.png"))
            plt.savefig(os.path.join(output_dir, img_name + "_markingall_cls.png"))
        elif args.display_method == "sub_marking_all_vid_idx" :
            print("save " + os.path.join(output_dir, img_name + "_markingall_vid.png"))
            plt.savefig(os.path.join(output_dir, img_name + "_markingall_vid.png"))
    
    #? 'detail_with_vid_idxs' ver.
    #? savefig for all vid idxs over several times
    if args.display_method == "detail_with_vid_idxs" :
        display_cls_list = args.detail_cls_list    #* classes desired to find video idxs
        
        assert args.num_vids_per_cls is not None, "current code works with only features that has fixed num videos per class"
        
        plt_texts = None
        for img_idx, i in enumerate(range(0, args.num_vids_per_cls, 4)) :
            end_idx = min(i + 4, args.num_vids_per_cls)
            display_vid_list = range(i, end_idx)   #* use subset of vid idxs to plt.text
            
            #* reset plt.text
            if plt_texts is not None :
                for txt in plt_texts :
                    txt.remove()
            
            #* plt.text
            plt_texts = finding_videos_in_display_list(plt, data_tsne, labels, display_cls_list, display_vid_list)
            
            #* save
            plt.savefig(os.path.join(output_dir, img_name + "_detail_ver{}.png".format(img_idx)))
            
        
          
    print("end...")
