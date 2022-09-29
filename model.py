import torch
import deepxde as dde
import numpy as np


class FEMFNN(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 input_sup,
                 input_inf,
                 num_nodes,
                 activation=torch.nn.SiLU(),
                 device='cpu'
                 ):
        super().__init__()
        self.input_sup = input_sup
        self.input_inf = input_inf
        self.device = device
        self.activation = activation

        self.nets = []

        self.nets.append(torch.nn.Linear(input_dim, num_nodes[0]))
        for i in range(len(num_nodes) - 1):
            self.nets.append(torch.nn.Linear(num_nodes[i], num_nodes[i + 1]))
        self.coefficient = torch.empty(output_dim, num_nodes[-1])

        self.nets = torch.nn.Sequential(*self.nets)

        self.to(device)

    def forward(self, x):
        # x = self._normalize(x, self.input_sup, self.input_inf)

        for net in self.nets:
            x = net(x)
            x = self.activation(x)

        return x


class FEMGAN:
    def __init__(self, net_u: FEMFNN, net_phi: FEMFNN, geometry: dde.geometry, size: int, rhs) -> None:
        self.net_u = net_u
        self.net_phi = net_phi
        self.geometry = geometry
        self.rhs = rhs
        self.size = size
        self.boundary_size = 1000
        self.volumn = self.geometry._r2 * np.pi
        self.surface_volumn = 2 * np.pi * self.geometry.radius

    def pre_train(self):
        pass

    def cal_loss(self):
        # TODO: non-zero boundary conditions
        points = self.geometry.random_points(self.size)
        points = torch.tensor(points, requires_grad=True, dtype=torch.float32, device=self.net_u.device)

        value_rhs = self.rhs(points)
        value_phi = self.net_phi(points)
        rhs = self.volumn * torch.mean(value_rhs * value_phi, dim=0).squeeze()
        # print(rhs.max())

        # 计算边界上的值
        boundary_points = self.geometry.random_boundary_points(self.boundary_size)
        boundary_normal = self.geometry.boundary_normal(boundary_points)
        boundary_points = torch.tensor(boundary_points, requires_grad=True, dtype=torch.float32, device=self.net_u.device)
        boundary_normal = torch.tensor(boundary_normal, requires_grad=True, dtype=torch.float32, device=self.net_u.device)
        
        f_s = self.net_u(points)

        gradient_phi = []
        for i in range(rhs.size()[-1]):
            grad_phi_i = torch.autograd.grad(
                value_phi[..., i:i + 1],
                points,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones_like(value_phi[..., i:i + 1])
            )[0]
            gradient_phi.append(grad_phi_i)

        gradient_f_s = []
        for i in range(f_s.size()[-1]):
            grad_f_i = torch.autograd.grad(
                f_s[..., i:i + 1],
                points,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones_like(f_s[..., i:i + 1])
            )[0]
            gradient_f_s.append(grad_f_i)


        bdy_f_s = self.net_u(boundary_points)
        bdy_phi = self.net_phi(boundary_points)

        gradient_bdy_f_s = []
        for i in range(bdy_f_s.size()[-1]):
            grad_bdy_f_i = torch.autograd.grad(
                bdy_f_s[..., i:i + 1],
                boundary_points,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones_like(bdy_f_s[..., i:i + 1])
            )[0]
            gradient_bdy_f_s.append((grad_bdy_f_i * boundary_normal).sum(dim=-1))


        A = torch.empty([2 * rhs.shape[0], f_s.size()[-1]], dtype=torch.float32, device=self.net_u.device)

        for i in range(A.shape[0]//2):
            for j in range(A.shape[1]):
                A[i,j] = self.volumn * torch.mean( (gradient_phi[i] * gradient_f_s[j]).sum(dim=-1) )
                A[i,j] -= torch.mean(bdy_phi[..., i:i+1] * gradient_bdy_f_s[j]) * self.surface_volumn
                # print(A[i, j])
            
        for i in range(A.shape[0]//2):
            for j in range(A.shape[1]):
                A[i+A.shape[0]//2,j] = torch.mean(
                    bdy_f_s[..., j:j+1] * bdy_phi[..., i:i+1]
                    ) * self.surface_volumn
                # print(bdy_f_s[..., j:j+1])
                # print(bdy_phi[..., i:i+1])
                # print(A[i,j])
        
        bdy_rhs = torch.zeros_like(rhs)
        for i in range(bdy_rhs.size()[-1]):
            bdy_rhs[i] = torch.mean(bdy_phi[..., i:i+1]) * self.surface_volumn
        
        all_rhs = torch.concat([rhs, bdy_rhs])

        np_A = A.detach().to('cpu').numpy()
        np_rhs = all_rhs.detach().to('cpu').numpy().reshape([-1,1])

        # print(np_A)
        # print(np_rhs)

        coef = np.linalg.inv(np_A.T.dot(np_A)).dot(np_A.T).dot(np_rhs)
        # coef = np.linalg.pinv(np_A).dot(np_rhs)
        # print(coef)

        coef = torch.tensor(coef, dtype=torch.float32, device=self.net_u.device)
        self.net_u.coefficient = coef.reshape([1,-1])

        loss = torch.mean((A @ coef - rhs) ** 2)
        
        return loss

    def train(self, optimizer_u, optimizer_phi, epochs=100):
        for i in range(epochs):
            optimizer_u.zero_grad()
            optimizer_phi.zero_grad()

            loss_u = self.cal_loss()
            loss_phi = - loss_u

            loss_u.backward(retain_graph=True)
            loss_phi.backward()

            optimizer_u.step()
            optimizer_phi.step()

            print("epoch {epoch}: loss={loss}".format(epoch=i, loss=loss_u))

    def evaluate(self, x):
        self.cal_loss()

        x = torch.tensor(x, dtype=torch.float32, device=self.net_u.device)
        pre_y = self.net_u(x)
        # print(pre_y.detach().numpy())
        y = torch.sum(self.net_u.coefficient * pre_y, dim=-1)
        return y
