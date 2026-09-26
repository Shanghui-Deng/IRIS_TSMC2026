import torch
import torch.nn as nn
import torch.nn.functional as F
from click.core import batch
from torch.nn.functional import normalize

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(x_dim, h_dim[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(h_dim) - 1):
            layers.append(nn.Linear(h_dim[i], h_dim[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(h_dim[-1], z_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.encoder(x))

class Decoder(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(z_dim, h_dim[-1]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(h_dim) - 1):
            layers.append(nn.Linear(h_dim[i], h_dim[i - 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(h_dim[-1], x_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

class CycleProjector(nn.Module):
    def __init__(self,input_dimx,input_dimy):
        super(CycleProjector, self).__init__()
        self.projectorxy = nn.Sequential(nn.Linear(input_dimx, input_dimy))
        self.projectoryx = nn.Sequential(nn.Linear(input_dimy, input_dimx))
    def forward(self, x,y):
        return self.projectorxy(x), self.projectoryx(y)

class Projector(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, out_dim),)
    def forward(self, z):
        return self.projector(z)



class MLP(nn.Module):
    def __init__(self, code_dim, layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.layers = layers
        self.hidden = nn.ModuleList()

        for k in range(layers):
            linear_layer = nn.Linear(code_dim, code_dim, bias=False)
            self.hidden.append(linear_layer)

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z
class DeepMulti(nn.Module):
    def __init__(self, input_dims,h_dim = 0,z_dim = 0,with_lowrank = True,cluster_num = 10):
        super(DeepMulti, self).__init__()

        self.z_dim = z_dim
        self.view_num = len(input_dims)
        self.with_lowrank = with_lowrank

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mpls = nn.ModuleList()
        self.cross_mlps = nn.ModuleList()
        self.cross_atten1 = nn.MultiheadAttention(z_dim,num_heads=1)
        self.cross_atten2 = nn.MultiheadAttention(z_dim, num_heads=1)
        for v in range(self.view_num):
            self.encoders.append(Encoder(input_dims[v], [h_dim, h_dim], z_dim))
            self.decoders.append(Decoder(z_dim, [h_dim, h_dim], input_dims[v]))
            if (self.with_lowrank):
                self.mpls.append(MLP(z_dim, 2))
                self.cross_mlps.append(MLP(z_dim,2))

    def MES_loss(self,X,X_Rec):
        X = X.to(torch.device("cuda"))
        X_Rec = X_Rec.to(torch.device("cuda"))
        return F.mse_loss(X,X_Rec)

    def forward(self, multiviewdata):
        recs = []
        zs = []
        for v in range(self.view_num):
            z_item = self.encoders[v](multiviewdata[v])
            if self.with_lowrank:
                h_item = self.mpls[v](z_item)
            rec = self.decoders[v](h_item)
            recs.append(rec)
            zs.append(z_item)
        cross_att1,_ = self.cross_atten1(zs[0],zs[1],zs[1])
        cross_att1 = F.normalize(cross_att1)
        cross_att2,_ = self.cross_atten2(zs[1],zs[0],zs[0])
        cross_att2 = F.normalize(cross_att2)
        fusion = torch.concat(
            ((cross_att1+zs[0])/2,(cross_att2+zs[1])/2),dim=1
        )

        return recs,zs,fusion,cross_att1,cross_att2

