import torch 
import torch.nn as nn
from tensornetwork import TensorRing, BrickTube
import torchquantum as tq
from transformers import BertModel
import math

class VQC(nn.Module):
    def __init__(self, n_wires, n_layers = 10):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.n_params = 3 * n_wires * n_layers 
        self.entangle_pairs = [(i, i + 1) for i in range(0, n_wires - 1, 2)]
        if n_wires > 2:
            self.entangle_pairs.append((n_wires - 1, 0))  
       
    def forward(self, params):
        B = params.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B, device=params.device)
        #Подаем на вход вакуумное состояние |0...0>
        qdev.reset_states(bsz=B)

        for layer in range(self.n_layers):
                # Запутывание 
                for w1, w2 in self.entangle_pairs:
                    tq.CNOT()(qdev, wires=[w1, w2])

                # Параметризованные вращения: =RX → RY → RZ на каждый кубит
                for i in range(self.n_wires):
                    base_idx = layer * (3 * self.n_wires) + 3 * i
                    theta_x = params[:, base_idx + 0]
                    theta_y = params[:, base_idx + 1]
                    theta_z = params[:, base_idx + 2]

                    tq.RX(has_params=False)(qdev, wires=i, params=theta_x)
                    tq.RY(has_params=False)(qdev, wires=i, params=theta_y)
                    tq.RZ(has_params=False)(qdev, wires=i, params=theta_z)

        #Измерение
        states = qdev.get_states_1d()  # комплекснозначный тензор
        probs = torch.abs(states) ** 2  # [B, 2**n_wires]
        return  probs
    
class TensorMeta_VQC(nn.Module):
    def __init__(self, num_class=2, bert_model_name="huawei-noah/TinyBERT_General_4L_312D", bert_dim=312, n_layers = 10):
        super().__init__()
        self.n_wires = math.ceil(math.log2(num_class))
        self.n_params = 3 * self.n_wires * n_layers 

        self.bert = BertModel.from_pretrained(bert_model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.tn = TensorRing(input_dim=bert_dim, output_dim=self.n_params, rank = 4, leg_dim = 2)
        #self.tn = BrickTube(bond_dim=3, input_dim=bert_dim, output_dim=self.n_params, n_layers=2)
        self.vqc = VQC(n_wires = self.n_wires)
        self.proj = nn.Linear(2**self.n_wires, num_class)
        

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = bert_outputs.last_hidden_state[:, 0, :]  # [B, bert_dim]
        theta = self.tn(cls_emb)  # [B, 3 * n_wires]
        vqc_out = self.vqc(theta)  # [B, n_wires]
        out = self.proj(vqc_out)
        return out
    
