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
        
        # Метки входа: [B, C, i0, i1, ..., i_{Q-1}]
        self.input_labels = list(range(self.Q + 2))  # 0,1 — batch/channel; 2..Q+1 — входные индексы
        self.output_labels = list(range(self.Q + 2, 2 * self.Q + 2))  # j0, j1, ..., j_{Q-1} — выходные индексы

        #Метки для внутренних сверток между ядрами
        bond_labels = []
        current = 2 * self.Q + 2
        for _ in range(self.Q + 1):
            bond_labels.append(current)
            current += 1

        #Метки для каждого ядра
        self.core_labels = []
        for i in range(self.Q):
            if i == self.Q - 1:
                labels = [self.input_labels[2 + i], bond_labels[i], bond_labels[0], self.output_labels[i]]
            else:
                labels = [self.input_labels[2 + i], bond_labels[i], bond_labels[i + 1], self.output_labels[i]]
            self.core_labels.append(labels)
    
    def forward(self, x):
        B, C, N = x.shape
        #Делаем паддинг нулями для нужной размерности. latent_dim >= N всегда
        x = F.pad(x, (0, self.latent_dim - N), "constant", 0) 
        x = x.view(B, C, *[self.leg_dim] * self.Q)

        #Подготовка аргументов для einsum
        args = [x, self.input_labels]
        for core, labels in zip(self.cores, self.core_labels):
            args += [core, labels]

        #Выходные метки: batch, channel, выходные физ. индексы
        result_labels = [0, 1] + self.output_labels

        out = torch.einsum(*args, result_labels)
        out = out.reshape(B, C, -1)
        out = out[:, :, :self.output_dim]
        return out



