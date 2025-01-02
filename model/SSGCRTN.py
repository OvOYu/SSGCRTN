import torch.nn as nn
import torch
import torch.nn.functional as F
import math
# import numpy as np

class TimeAttention(nn.Module):
    def __init__(self,
                 outfea,
                 d):
        super(TimeAttention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.d = d

    def forward(self,
                x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        return value

# 图卷积
class gconv_hyper(nn.Module):
    def __init__(self):
        super(gconv_hyper, self).__init__()

    def forward(self,
                x,
                A):

        if len(A.shape) == 2:
            try:
                x = torch.einsum('wv,bvc->bwc',
                                 (A, x))
            except:
                print("x shape:", x.shape)
                print("A shape:", A.shape)
                exit(-1)
        elif len(A.shape) == 3:
            if x.shape[0] == A.shape[0] and len(x.shape) == len(A.shape):
                x = torch.einsum('bwv,bvc->bwc',
                                 (A, x))
            else:
                batch_size, head_num = x.shape[0], A.shape[0]

                A = A.unsqueeze(0).repeat(batch_size, 1, 1, 1)

                x = torch.einsum('bnwv,bnvc->bnwc', (
                    A, x))

        elif len(A.shape) == 4:
            x = torch.einsum('bnwv,bnvc->bnwc', (
                A, x))

        return x.contiguous()


class GCNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 gdep=2):
        super(GCNLayer, self).__init__()
        self.gdep = gdep
        self.gconv_preA = gconv_hyper()
        self.mlp = nn.Linear((self.gdep + 1) * in_dim, out_dim)

    def forward(self,
                adj,
                x):
        h = x
        out = [h]

        for _ in range(self.gdep):

            h = self.gconv_preA(h, adj)  # (16,170,64)

            out.append(h)

        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho



class GCRNCell(nn.Module):
    def __init__(self,
                 node_num,
                 in_dim,  # 1
                 out_dim):  # 64
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.gz = GCNLayer(in_dim + out_dim, in_dim + out_dim)
        self.gr = GCNLayer(in_dim + out_dim, in_dim + out_dim)
        self.gc = GCNLayer(in_dim + out_dim, in_dim + out_dim)
        self.hidden_dim = out_dim
        self.mlp = nn.Linear(in_dim + out_dim, out_dim)
        self.a = nn.Parameter(torch.rand(1), requires_grad=True)
    def self_attn_func(self,
                       x):
        x = x.transpose(0, 1)
        x_attn_out, _ = self.self_attn(x, x, x)
        x_attn_out = x_attn_out.transpose(0, 1)
        return x_attn_out

    def gcn_forward(self,
                    x_hidden,
                    predefine_A,
                    add_weight,
                    gcn_layers,
                    E1=None,
                    E2=None):
        res = 0
        if add_weight['w_pre'] > 0.00001:  # 进入


            predefine_gcn = sum([gcn_layers(each, x_hidden) for each in predefine_A])  # (16,170,64)


        if add_weight['w_adp'] > 0.00001:
            [_, head_num, _] = E1.shape
            sqrt_d = math.sqrt(E1.shape[-1])

            A = F.relu(torch.einsum('nhc,nvc->nhv',
                                    (E1.transpose(0, 1), E2.transpose(0, 1)))) / sqrt_d

            A = F.softmax(A, dim=-1)  # (8,170,170)


            agcn_res = gcn_layers(A, x_hidden.unsqueeze(1).repeat(1, head_num, 1, 1)).mean(1)
            out_g = self.a * predefine_gcn + agcn_res*(1-self.a)
            res += out_g

        return res

    def init_hidden_state(self,
                          batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

    def forward(self,
                x,
                hidden_state,
                predefine_A,
                add_weight,
                E1=None,
                E2=None):
        '''
        :param input: batch_size * node_num * 1
        :param hidden_state: batch_size * node_num * hidden_dim
        :param predefine_A: node_num * node_num
        :param E1: node_num * agcn_head_num * (node_emb_dim // agcn_head_num)
        :param E2:
        :param batch_size:
        :return:
        '''
        state =  hidden_state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z = F.sigmoid(self.gcn_forward(input_and_state, predefine_A, add_weight, self.gz, E1,
                                       E2))


        r = F.sigmoid(self.gcn_forward(input_and_state, predefine_A, add_weight, self.gr, E1,
                                       E2))


        c = F.tanh(
            self.gcn_forward(torch.mul(r, input_and_state), predefine_A, add_weight, self.gc, E1, E2))  # (16,170,64)

        hidden_state = torch.mul(1 - z, input_and_state) + torch.mul(z, c)  # (16,170,64)

        hidden_state = self.mlp(hidden_state)

        return hidden_state


class SSGCRN(nn.Module):
    def __init__(self,
                 node_num,
                 dim_in,  # 1
                 dim_out):  # 64
        super(SSGCRN, self).__init__()

        self.node_num = node_num

        self.input_dim = dim_in

        self.stfgrn_cells = GCRNCell(node_num, dim_in, dim_out)



    def forward(self,
                x,
                adj,
                add_weight,
                init_state,
                E1,
                E2):

        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []

        state = init_state
        inner_states = []
        for t in range(seq_length):
            state = self.stfgrn_cells(current_inputs[:, t, :, :],  state, adj,add_weight, E1, E2)
            inner_states.append(state)
        output_hidden.append(state)
        current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden

    def init_hidden(self,
                    batch_size):
        init_states = self.stfgrn_cells.init_hidden_state(batch_size)
        return init_states



class STIM(nn.Module):
    def __init__(self,
                 node_num,  # 170
                 dim_in,  # 1
                 dim_out):  # 64
        super(STIM, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.dim_out = dim_out
        self.SSGCRNS = nn.ModuleList()

        self.SSGCRNS.append(SSGCRN(node_num, dim_in, dim_out))
        for _ in range(2):
            self.SSGCRNS.append(SSGCRN(node_num, dim_in, dim_out))

    def forward(self,
                x,  # (64,12,170,1)
                adj,
                add_weight,
                E1, # (170,8,16)
                E2 # (170,8,16)
                ):
        init_state_R = self.SSGCRNS[0].init_hidden(x.shape[0])  # (64,170,64)
        init_state_L = self.SSGCRNS[1].init_hidden(x.shape[0])  # (64,170,64)

        # print("adj:", adj.shape)
        h_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.dim_out * 2).to(
            x.device)  # (64,12,170,128)  初始化一个输出（状态）矩阵
        out1 = self.SSGCRNS[0](x, adj, add_weight, init_state_R, E1, E2 )[0]  # (64,12,170,64)
        out2 = self.SSGCRNS[1](torch.flip(x, [1]), adj, add_weight, init_state_L, E1, E2)[0]  # (64,12,170,64)

        h_out[:, :, :, :self.dim_out] = out1
        h_out[:, :, :, self.dim_out:] = out2
        return h_out  # (64,12,170,128)

class SSGCRTN(nn.Module):
    def __init__(self,
                 num_nodes,   # 170
                 input_dim,   # 1
                 rnn_units,   # 64
                 output_dim,   # 1
                 horizon,   # 12
                 at_filter,   # 16
                 w_pre, # 0.1
                 w_adp, # 0.9
                 node_emb_dim, # 128
                 agcn_head_num):  # 2
        super(SSGCRTN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon


        self.STIM = STIM(self.num_node, self.input_dim, self.hidden_dim)



        self.timeAtt = TimeAttention(self.hidden_dim * 2, at_filter)
        self.out_emd = nn.Linear(self.hidden_dim * 2, output_dim)


        self.add_weight = {"w_pre": w_pre, "w_adp": w_adp}
        self.spatial_dependency_layer = GCRNCell(self.num_node,self.hidden_dim, self.hidden_dim)


        if self.add_weight['w_adp'] > 0.00001:
            self.head_num = agcn_head_num
            self.E1 = nn.Parameter(
                torch.randn(num_nodes, agcn_head_num, node_emb_dim // agcn_head_num),
                requires_grad=True)
            self.E2 = nn.Parameter(
                torch.randn(num_nodes, agcn_head_num, node_emb_dim // agcn_head_num),
                requires_grad=True)
        else:
            self.E1, self.E2 = None, None



    def forward(self,
                source,
                graph):

        source = source.transpose(1, 3).transpose(2, 3)


        output = self.STIM(source,  graph, self.add_weight, self.E1, self.E2)


        trans = self.timeAtt(output)


        out = self.out_emd(trans).transpose(1, 2)

        return out.view(trans.shape[0], trans.shape[2], -1)