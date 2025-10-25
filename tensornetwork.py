import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def log_base(x, base):
    return math.log(x) / math.log(base)

class TensorRing(nn.Module):
    def __init__(self, input_dim, output_dim, rank, leg_dim):
        '''             
      -------------------
      |                 |
      | |  |       |  | |
      --*--*--...--*--*--     
        |  |       |  |

        input_dim - размерность входного вектора
        output_dim - размерность выходного вектора
        rank - размерность "внутренних" индексов 
        leg_dim - размерность "внешних" индексов
         * - тензорное ядро ; |,--  - индексы тензоров
        '''
        super().__init__()
        self.rank = rank
        self.leg_dim = leg_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        dim = max(input_dim, output_dim)
        #Количество "внешних" ножек
        self.Q = math.ceil(log_base(dim, leg_dim)) 
        #Округляем входную размерность до rank2^Q
        self.latent_dim = leg_dim**self.Q  
        self.cores = nn.ParameterList([
                nn.Parameter(torch.empty(leg_dim, rank, rank, leg_dim))
                for _ in range(self.Q)
            ])
        
        #Инициализация
        var = 1 /(rank*leg_dim)
        var *= (self.latent_dim / input_dim) ** (1 / self.Q)
        b = math.sqrt(3 * var)
        for core in self.cores:
            nn.init.uniform_(core, -b, b)
        
        # Метки входа: [B, i0, i1, ..., i_{Q-1}]
        self.input_labels = list(range(self.Q + 1))  # 0,1 — batch/channel; 2..Q+1 — входные индексы
        self.output_labels = list(range(self.Q + 1, 2 * self.Q + 1))  # j0, j1, ..., j_{Q-1} — выходные индексы

        #Метки для внутренних сверток между ядрами
        bond_labels = []
        current = 2 * self.Q + 1
        for _ in range(self.Q + 1):
            bond_labels.append(current)
            current += 1

        #Метки для каждого ядра
        self.core_labels = []
        for i in range(self.Q):
            if i == self.Q - 1:
                labels = [self.input_labels[1 + i], bond_labels[i], bond_labels[0], self.output_labels[i]]
            else:
                labels = [self.input_labels[1 + i], bond_labels[i], bond_labels[i + 1], self.output_labels[i]]
            self.core_labels.append(labels)
    
    def forward(self, x):
        B, N = x.shape
        #Делаем паддинг нулями для нужной размерности. latent_dim >= N всегда
        x = F.pad(x, (0, self.latent_dim - N), "constant", 0) 
        x = x.view(B, *[self.leg_dim] * self.Q)

        #Подготовка аргументов для einsum
        args = [x, self.input_labels]
        for core, labels in zip(self.cores, self.core_labels):
            args += [core, labels]

        #Выходные метки: batch, выходные индексы
        result_labels = [0] + self.output_labels

        out = torch.einsum(*args, result_labels)
        out = out.reshape(B, -1)
        out = out[:, :self.output_dim]
        return out

class BrickTube(nn.Module):
    def __init__(self, bond_dim, input_dim, output_dim, n_layers):
        super().__init__()
        dim = max(input_dim, output_dim)
        Q = math.ceil(log_base(dim, bond_dim))
        self.Q = Q + Q%2
        self.bindim = bond_dim**self.Q
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.static_rank = self.Q + 1
        self.base_labels = tuple(range(self.static_rank))

        total_cores = n_layers * self.Q
        self.cores = nn.ParameterList(
            [
                nn.Parameter(torch.empty(bond_dim, bond_dim, bond_dim, bond_dim))
                for _ in range(total_cores)
            ]
        )
        var = (1 / self.bond_dim) ** 2
        p_a = torch.nn.init.calculate_gain('linear')
        var *= (self.bindim / input_dim) ** (1 / total_cores)
        var *= (p_a) ** (2 / total_cores)
        b = math.sqrt(3 * var)
        for core in self.cores:
            nn.init.uniform_(core, -b, b)
        self.contraction_pairs1 = [(i, i + 1) for i in range(0, self.Q, 2)]
        self.contraction_pairs2 = [(i, (i + 1) % self.Q) for i in range(1, self.Q, 2)]
        self.n_layers = n_layers
        self.cores_per_layer = self.Q 
        max_base_label = max(self.base_labels)
        current_max = max_base_label + 1
        self.t_labels_offset1 = []
        for x_idx, x_idx1 in self.contraction_pairs1:
            old0, old1 = x_idx + 1, x_idx1 + 1
            new0, new1 = current_max, current_max + 1
            current_max += 2
            self.t_labels_offset1.append((old0, old1, new0, new1))

        self.t_labels_offset2 = []
        for x_idx, x_idx1 in self.contraction_pairs2:
            old0, old1 = x_idx + 1, x_idx1 + 1
            new0, new1 = current_max, current_max + 1
            current_max += 2
            self.t_labels_offset2.append((old0, old1, new0, new1))

    def brickconv(self, tensor, cores, t_labels_offset):
        inputs = [tensor, self.base_labels]
        labels = list(self.base_labels)
        for i, (x0, x1, n0, n1) in enumerate(t_labels_offset):
            inputs.extend([cores[i], [x0, x1, n0, n1]])
            labels[x0] = n0
            labels[x1] = n1
        return torch.einsum(*inputs, labels)

    def brickwall_layer_forward(self, x, cores1, cores2):
        x = self.brickconv(x, cores1, self.t_labels_offset1)
        x = self.brickconv(x, cores2, self.t_labels_offset2)
        return x

    def forward(self, x):
        B, N = x.shape
        x = F.pad(x, (0, self.bindim - N), "constant", 0)
        x = x.view(B, *([self.bond_dim] * self.Q)).requires_grad_(True)
        for layer in range(self.n_layers):
            start_idx = layer * self.cores_per_layer
            cores1 = self.cores[start_idx : start_idx + len(self.contraction_pairs1)]
            cores2 = self.cores[
                start_idx + len(self.contraction_pairs1) : start_idx + self.cores_per_layer
            ]

            x = self.brickwall_layer_forward(x, cores1, cores2)
         
        x = x.reshape(B, -1)
        x = x[:, : self.output_dim]

        return x

