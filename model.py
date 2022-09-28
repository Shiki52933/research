import torch
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

    def _normalize(self, x, input_sup, input_inf):
        x = (x - input_inf) / (input_sup - input_inf)
        x = 2 * x - 1
        return x

    def forward(self, x):
        x = self._normalize(x, self.input_sup, self.input_inf)

        for net in self.nets:
            x = net(x)
            x = self.activation(x)

        return x


class FEMGAN:
    def __init__(self, net_u: FEMFNN, net_phi: FEMFNN, geometry, size: int, rhs) -> None:
        self.net_u = net_u
        self.net_phi = net_phi
        self.geometry = geometry
        self.rhs = rhs
        self.size = size

    def cal_loss(self):
        # TODO: non-zero boundary conditions
        points = self.geometry.random_points(self.size)
        points = torch.tensor(points, requires_grad=True, dtype=torch.float32, device=self.net_u.device)

        value_rhs = self.rhs(points)
        value_phi = self.net_phi(points)
        rhs = torch.mean(value_rhs * value_phi, dim=0).squeeze()
        # print(rhs.max())

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

        A = torch.empty([rhs.shape[0], f_s.size()[-1]], dtype=torch.float32, device=self.net_u.device)

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i, j] = torch.mean( (gradient_phi[i] * gradient_f_s[j]).sum(dim=-1) )
                # print(A[i, j])

        np_A = A.detach().to('cpu').numpy()
        np_rhs = rhs.detach().to('cpu').numpy()

        # print(np_A)
        # print(np_rhs)

        coef = np.linalg.pinv(np_A).dot(np_rhs)
        # print(coef)

        coef = torch.tensor(coef, dtype=torch.float32, device=self.net_u.device)
        self.net_u.coefficient = coef.reshape([1,-1])

        loss = torch.mean(A @ coef - rhs) ** 2

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
