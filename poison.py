import numpy as np
import torch

import model
import sympy
import deepxde as dde

sup = [1., 1.]
inf = [-1., -1.]

input_sup = torch.tensor(np.array([sup]), dtype=torch.float32, device='cpu')
input_inf = torch.tensor(np.array([inf]), dtype=torch.float32, device='cpu')

def rhs(x):
    return -4*torch.ones([x.size()[0], 1])


geo = dde.geometry.Disk([0,0], 1)

net_u = model.FEMFNN(2, 1, input_sup=input_sup, input_inf=input_inf, num_nodes=[20]*4, activation=torch.nn.ReLU())
net_phi = model.FEMFNN(2, 1, input_sup=input_sup, input_inf=input_inf, num_nodes=[20]*4, activation=torch.nn.SiLU())
gan = model.FEMGAN(net_u=net_u, net_phi=net_phi, geometry=geo, size=1000, rhs=rhs)

optimizer_u = torch.optim.Adam(params=net_u.parameters(), lr=1e-4)
optimizer_phi = torch.optim.Adam(params=net_phi.parameters(), lr=1e-2)

gan.train(optimizer_u, optimizer_phi, 10)

x = geo.random_points(10)
y = gan.evaluate(x).detach().numpy()

# print(x)
# print(y)
# print(net_u.coefficient)

res = x[:,0] ** 2 + x[:,1] ** 2 - y
res = np.abs(res)
print(res.max())