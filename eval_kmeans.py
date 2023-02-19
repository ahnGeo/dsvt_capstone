from sklearn.cluster import KMeans
import pickle
import argparse
import torch
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
