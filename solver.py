import torch
import numpy as np
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from all_in_one import tsne,kmeans
from clusteringPerformance import clusteringMetrics
from sklearn.cluster import KMeans
import os
from itertools import chain
import matplotlib.pyplot as plt
# from loss import MultiViewContrastiveLoss,ContrastiveLoss,Loss
import sys
import torch_clustering
import torch.nn.functional as F

class Solver():
    def __init__(self, args,model,train_loader = None,test_loader = None,view_num=3,num_clusters = 20,epochs = 0,pretrain_epoch=0,lr = 0.0008,whenprint = None):
        self.model = model
        self.view_num = view_num
        self.epochs = epochs
        self.pretrain_epoch = pretrain_epoch
        self.num_clusters = num_clusters

        self.data_loader  = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda")

        self.args = args

        self.model.to(self.device)

        self.whenprint = whenprint

        self.optimizer = torch.optim.Adam(chain(self.model.parameters()), lr=lr, weight_decay=args.weight_decay)
        self.acc = 0
        self.nmi = 0
        self.ari = 0
        self.fscore = 0
        self.precision = 0
        self.recall = 0
        self.purity = 0
        self.ami = 0
        if not os.path.exists(r'./results/'+ self.args.data_name):
            os.makedirs(r'./results/'+ self.args.data_name)

    def vaild(self,epoch,datametric,mode):
        if epoch % 1 == 0:
            self.model.eval()
            with torch.no_grad():
                for x, y, idx in self.test_loader:
                    for v in range(self.view_num):
                        x[v] = x[v].to(self.device)
                    # recs, zs = self.model.forward_fine(x)
                    _, zs,fusion,cross_att1,cross_att2 = self.model(x)
                    label = np.array(y)
                    kwargs = {
                        'metric': 'cosine',
                        'distributed': False,
                        'random_state': 0,
                        'n_clusters': self.num_clusters,
                        'verbose': False
                    }
                    clustering_model_fusion = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4,**kwargs)
                    psedo_labels = clustering_model_fusion.fit_predict(fusion.to(dtype=torch.float64))
                    ACC, NMI, ARI, Purity,  F_score, Precision, Recall,AMI = clusteringMetrics(label, psedo_labels.cpu().numpy())
                    info = {"epoch": epoch, "ACC": np.around(ACC, 4), "NMI": np.around(NMI, 4),
                            "ARI": np.around(ARI, 4),"Purity":np.round(Purity,4), "F-score": np.around(F_score, 4),
                            "Precision": np.around(Precision, 4), "Recall": np.around(Recall, 4),"AMI":np.round(AMI,4)}
                    # print(info)
                    if self.acc < ACC and mode=='train':
                        self.acc = ACC
                        self.nmi = NMI
                        self.ari = ARI
                        self.purity = Purity
                        self.fscore = F_score
                        self.precision = Precision
                        self.recall = Recall
                        self.ami = AMI
                        self.epoch = epoch


    def pretrain(self):
        datametric = []

        for epoch in range(self.pretrain_epoch):
            log = {
                'total_loss': [],
            }
            for x, y, idx in self.data_loader :

                self.model.train()
                self.optimizer.zero_grad()

                for v in range(self.view_num):
                    x[v] = x[v].to(self.device)
                recs,zs,_fusion,cross_att1,cross_att2= self.model(x)
                loss = 0
                for v in range(self.view_num):
                    nuclear_norms = []
                    for layer in self.model.cross_mlps[v].hidden:
                        if hasattr(layer, 'weight'):
                            weight = layer.weight
                            nuclear_norm = torch.linalg.matrix_norm(weight, ord='nuc')
                            nuclear_norms.append(nuclear_norm)
                    total_nuclear_norm = sum(nuclear_norms)
                    loss += self.model.MES_loss(recs[v],x[v])
                    loss = loss+total_nuclear_norm
                loss.backward()
                self.optimizer.step()

                log['total_loss'].append(loss.item())
            print(f"Epoch {epoch} loss: {np.mean(log['total_loss'])}")

            # self.vaild(epoch, datametric,mode='pre')

    def train_one_epoch(self):
        datametric = []

        for epoch in range(self.epochs):
            log = {
                'total_loss': [],
            }
            for x, y, idx in self.data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                for v in range(self.view_num):
                    x[v] = x[v].to(self.device)
                recs,zs,fusion,cross_att1,cross_att2= self.model(x)
                loss = 0
                for v in range(self.view_num):
                    nuclear_norms = []
                    for layer in self.model.cross_mlps[v].hidden:
                        if hasattr(layer, 'weight'):
                            weight = layer.weight
                            nuclear_norm = torch.linalg.matrix_norm(weight, ord='nuc')
                            nuclear_norms.append(nuclear_norm)
                    total_nuclear_norm = sum(nuclear_norms)
                    loss += self.model.MES_loss(recs[v], x[v])
                    loss += total_nuclear_norm
                sim_loss_1 = self.model.MES_loss(zs[0],cross_att1)
                sim_loss_2 = self.model.MES_loss(zs[1],cross_att2)
                loss += sim_loss_1
                loss += sim_loss_2
                loss.backward()
                self.optimizer.step()
                log['total_loss'].append(loss.item())
            print(f"Epoch {epoch} loss: {np.mean(log['total_loss'])}")

            self.vaild(epoch, datametric,mode='train')
        info = {"ACC": np.around(self.acc, 4), "NMI": np.around(self.nmi, 4),
                "ARI": np.around(self.ari, 4), "Purity":np.round(self.purity,4), "F-score": np.around(self.fscore, 4),
                "Precision": np.around(self.precision, 4), "Recall": np.around(self.recall, 4),"AMI": np.around(self.ami, 4)}
        print(info)

