from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

feature_path = ["/data/geo123/features/timesformer-k400-kth-fe-8f.pkl",
                "/data/geo123/features/timesformer-sthv2-kth-fe-8f.pkl",
                "/data/geo123/features/outputs/kth/cut-one/testfeat.pkl",
                "/data/geo123/mmaction2/work_dirs/feature/tsm_k400_kth_noft_feature_8f.pkl",
                "/data/geo123/mmaction2/work_dirs/feature/tsm_k400_kth_noft_feature_16f.pkl",
                "/data/geo123/mmaction2/work_dirs/feature/tsm_sthv2_kth_noft_feature_8f.pkl",
                "/data/geo123/mmaction2/work_dirs/feature/tsm_sthv2_kth_noft_feature_16f.pkl"
]
label_path = "/data/geo123/features/datasets/annotations/kth_fe_videos.txt"

with open(label_path, 'r') as f :
    label = f.readlines()
    label = [label[i].strip('\n').split()[1] for i in range(len(label))]
    label = np.array(label)
    
for path in feature_path :
    with open(path, 'rb') as f :
        data = pickle.load(f)
        data = np.array(data)

    # print("data shape : ", data.shape)
    # print("label shape : ", label.shape)

    X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2, stratify=label, random_state=77, shuffle=True)

    # print("train, val shape : ", X_train.shape, X_val.shape)
    # print("y_val : ", y_val[:10])
    
    mlp = MLPClassifier(hidden_layer_sizes=(), solver='adam', max_iter=10000, learning_rate_init=1e-3, random_state=77, verbose=False, batch_size=64)
    mlp.fit(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn.fit(X_train, y_train)

    mlp_pred = mlp.predict(X_val)
    knn_pred = knn.predict(X_val)

    mlp_eval = (mlp_pred == y_val).tolist()
    knn_eval = (knn_pred == y_val).tolist()
    
    mlp_acc = mlp_eval.count(True) / len(mlp_eval)
    knn_acc = knn_eval.count(True) / len(knn_eval)

    print("path : ", path)
    print("linear acc : ", mlp_acc)
    print("knn acc : ", knn_acc)
    