import pickle 
import numpy as np


# load your dataset
with open("/data/ahngeo11/svt/outputs/ucf101/simple/testfeat.pkl", 'rb') as f :
    data = pickle.load(f)
print("data shape : ({}, {})".format(len(data), len(data[0])))
    
with open("/data/ahngeo11/svt/datasets/annotations/ucf101_val_split_1_videos_simple.txt", 'r') as f :
    label_idxs = f.readlines()
    label_idxs = [int(line.split()[1]) for line in label_idxs]
    
with open("/data/ahngeo11/mmaction2/data/ucf101/annotations/classInd.txt", 'r') as f :
    class_bind = f.readlines()
label_texts = dict()
for line in class_bind :
    idx, label = int(line.split()[0]), line.split()[1]
    label_texts[idx] = label

data = np.array(data)
data = np.reshape(data, (-1, 101, 768))

print(np.linalg.norm(data[59,2]-data[59,1]))