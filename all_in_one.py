
from torch.utils.data import Dataset
import scipy.io as scio
from sklearn import preprocessing
import random

#==============================================================================#
#===================================kmeans gpu================================#
#==============================================================================#
import torch
import numpy as np

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cuda')
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    # print(f'running k-means on {device}..')
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    dis_min = float('inf')
    initial_state_best = None
    for i in range(20):
        initial_state = initialize(X, num_clusters)
        dis = pairwise_distance_function(X, initial_state).sum()
        if dis < dis_min:
            dis_min = dis
            initial_state_best = initial_state

    initial_state = initial_state_best
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if iteration > 500:
            break
        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='cosine',
        device=torch.device('cuda')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    # print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


#==============================================================================#
#=================================  dataload   ================================#
#==============================================================================#
def loadData(data_name,x_label,y_label):
    #判断是否是mat或者npy文件

    if data_name.endswith('.mat'):
        data = scio.loadmat(data_name)
        datas = data[x_label]

        ground_true_label = data[y_label]
        ground_true_label = ground_true_label.flatten()
        return datas, ground_true_label

    elif data_name.endswith('.npy'):
        data = np.load(data_name)
        #同mat格式，在一个文件里有两个，feature和label，分别为X，Y
        datas = data[x_label]
        ground_true_label = data[y_label]
        ground_true_label = ground_true_label.flatten()
        return datas, ground_true_label

class MultiViewDataset(Dataset):

    def __init__(self, dataname,x_label,y_label):
        self.datas, self.gnd = loadData(dataname,x_label,y_label)
        self.v = self.datas.shape[1]

        # Normalize data
        scaler = preprocessing.MinMaxScaler()
        self.datas[0] = [scaler.fit_transform(feature).astype(np.float32) for feature in self.datas[0]]

    def __len__(self):
        return self.gnd.shape[0]

    def __getitem__(self, idx):
        data_list = [torch.from_numpy(self.datas[0][i][idx]) for i in range(self.v)]
        return data_list, torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx))

def MultiViewDatasetLoader(dataname,x_label,y_label):
    datas, ground_true_label = loadData(dataname,x_label,y_label)
    # views = datas[0].shape[0]
    views = 2
    input_num = datas[0][0].shape[0]
    multiviewdataset = MultiViewDataset(dataname,x_label,y_label)
    num_clusters = len(np.unique(ground_true_label))
    input_dims = [datas[0][v].shape[1] for v in range(views)]
    print(f"Data: {dataname}, Number of samples: {input_num}, Views: {views}, Clusters: {num_clusters}, Feature dimensions per view: {input_dims}")
    return multiviewdataset, input_num, views, num_clusters, input_dims

#==============================================================================#
#===================================yaml config================================#
#==============================================================================#
import os
import yaml
def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg
#==============================================================================#
#==============================================================================#
#==============================================================================#
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
   This program is to evaluate clustering performance

   Code author: Shide Du
   Email: shidedums@163.com
   Date: Dec 4, 2019.
'''
import torch
import torch.nn.functional as F
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import numpy.linalg as LA
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
# from sklearn.metrics.cluster.supervised import check_clusterings
# from sklearn.metrics.cluster.supervised import check_clusterings
import warnings
from scipy import sparse as sp
# from sklearn.utils.fixes import comb
from scipy.special import comb
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


def bestMap(y_pred, y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    label = np.zeros(y_pred.size)
    for i in range(y_pred.size):
        label[i] = ind[y_pred[i]][1]
    return label.astype(np.int64)


### K-means clustering
def KMeansClustering(features, gnd, clusterNum, randNum):
    kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                    random_state=randNum)
    estimator = kmeans.fit(features)
    clusters = estimator.labels_

    labels = np.zeros_like(clusters)
    for i in range(clusterNum):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    # sio.savemat('ALOI_idx.mat', {'idx': labels})
    # Return the preditive clustering label
    return labels


def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    # ind = linear_assignment(w.max() - w)
    indx_list = []
    for i in range(len(ind[0])):
        indx_list.append((ind[0][i], ind[1][i]))
    # return sum([w[i1, j1] for i1, j1 in ind]) * 1.0 / y_pred.size
    return sum([w[i1, j1] for (i1, j1) in indx_list]) * 1.0 / y_pred.size


def cluster_f(y_true, y_pred):
    N = len(y_true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(0, N):
        C1 = [y_true[n] for x in range(1, N - n)]
        C1 = np.array(C1)
        C2 = y_true[n + 1:]
        C2 = np.array(C2)
        Tn = (C1 == C2) * 1

        C3 = [y_pred[n] for x in range(1, N - n)]
        C3 = np.array(C3)
        C4 = y_pred[n + 1:]
        C4 = np.array(C4)
        Hn = (C3 == C4) * 1

        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(np.multiply(Tn, Hn))
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0;
    else:
        f = 2 * p * r / (p + r);
    return f, p, r


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins=range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct / len(y_pred)

    return Purity


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_pred = bestMap(labels_pred, labels_true)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall


### Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    y_pred = bestMap(predictiveLabel, trueLabel)
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, y_pred)

    # Normalized mutual information
    # NMI = metrics.v_measure_score(trueLabel, predictiveLabel)
    NMI = normalized_mutual_info_score(trueLabel, y_pred)

    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), y_pred.reshape(-1, 1))

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, y_pred)
    # ARI = rand_index_score(trueLabel, predictiveLabel)

    # Fscore, Precision, Recall = cluster_f(trueLabel, y_pred)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, y_pred)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall


### Report mean and std of 10 experiments
def StatisticClustering(features, gnd, clusterNum):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))

    # clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(gnd, predictiveLabel)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    # plt.scatter(features[:, 0], features[:, 2], c = predictiveLabel)
    # plt.savefig("Clustering_results.jpg")
    # plt.show()
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std


def StatisticClustering1(features, gnd):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 7
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    print("cluster number: ", clusterNum)
    for i in range(repNum):
        predictiveLabel = KMeansClustering(features, gnd, clusterNum, i)
        ACC, NMI, ARI = clusteringMetrics(gnd, predictiveLabel)
        ACCList[i] = ACC
        NMIList[i] = NMI
        ARIList[i] = ARI
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    return ACCmean_std, NMImean_std, ARImean_std


def spectral_clustering(points, k, gnd):
    W = similarity_function(points)
    # W = points
    Dn = np.diag(1 / np.power(np.sum(W, axis=1), -0.5))
    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)
    eigvecs = eigvecs.astype(float)
    indices = np.argsort(eigvals)[:k]
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])

    [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(k_smallest_eigenvectors, gnd, k)
    return [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def clustering_loss(features, gnd, clusterNum, alpha, device):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    PurityList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))
    FscoreList = np.zeros((repNum, 1))
    PrecisionList = np.zeros((repNum, 1))
    RecallList = np.zeros((repNum, 1))
    kl_loss = 0
    # clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1
    # print("cluster number: ", clusterNum)
    for i in range(repNum):
        kmeans = KMeans(n_clusters=clusterNum)
        kmeans.fit_predict(features.cpu().detach().squeeze().numpy())
        cluster_layer = torch.tensor(kmeans.cluster_centers_).to(device)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(features.squeeze().unsqueeze(1) - cluster_layer, 2), 2) / alpha)
        q = q.pow((alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        p = target_distribution(q)
        y_pred = q.cpu().detach().numpy().sargmax(1)
        # delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        # y_pred_last = y_pred
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(trueLabel=gnd, predictiveLabel=y_pred)

        ACCList[i] = ACC
        NMIList[i] = NMI
        PurityList[i] = Purity
        ARIList[i] = ARI
        FscoreList[i] = Fscore
        PrecisionList[i] = Precision
        RecallList[i] = Recall
        kl_loss += F.kl_div(q.log(), p)
        # print("ACC, NMI, ARI: ", ACC, NMI, ARI)
    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    Puritymean_std = np.around([np.mean(PurityList), np.std(PurityList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)
    Fscoremean_std = np.around([np.mean(FscoreList), np.std(FscoreList)], decimals=4)
    Precisionmean_std = np.around([np.mean(PrecisionList), np.std(PrecisionList)], decimals=4)
    Recallmean_std = np.around([np.mean(RecallList), np.std(RecallList)], decimals=4)
    # plt.scatter(features[:, 0], features[:, 2], c = predictiveLabel)
    # plt.savefig("Clustering_results.jpg")
    # plt.show()
    return ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std, kl_loss


### Real entrance to this program
if __name__ == '__main__':
    # Step 1: load data
    # features, gnd = loadData('./data/Yale_32x32.mat')
    # print("The size of data matrix is: ", features.shape)
    # gnd = gnd.flatten()
    # print("The size of data label is: ", gnd.shape)
    # clusterNum = 10
    # Print clustering results
    # [ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std] = StatisticClustering(
    #    features, gnd)
    # print("ACC, NMI, Purity, ARI, Fscore, Precision, Recall: ", ACCmean_std, NMImean_std, Puritymean_std, ARImean_std, Fscoremean_std, Precisionmean_std, Recallmean_std)
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gnd = np.array([1, 1, 1, 0, 0, 0])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    estimator = kmeans.fit(X)
    clusters = estimator.labels_
    label_pred = estimator.labels_
    labels = np.zeros_like(clusters)
    for i in range(2):
        mask = (clusters == i)
        labels[mask] = mode(gnd[mask])[0]
    print(labels)
    print(label_pred)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def tsne(H, Y,path,epoch):
    tsne = TSNE(random_state=0)
    view = tsne.fit_transform(H)
    n_class = len(np.unique(Y))
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(view[:, 0], view[:, 1], c=Y.squeeze(), s=16, cmap=plt.cm.get_cmap('jet', n_class))
    plt.colorbar(ticks=range(n_class + 1))

    plt.savefig(path+'/'+ str(epoch)+'.pdf')
