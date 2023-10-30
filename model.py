from .conv import *
from layers import ConvKB,TextCNN
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pack_sequence
import numpy as np
import torch.nn.functional as F
CUDA = torch.cuda.is_available()


class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda', temperature=0.5):
        super().__init__()
        #self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度

    def forward(self, batch_size, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 4020, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    

class TypeGAT(nn.Module):
    def __init__(self, num_e, num_r, relation_embeddings, out_dim,n_heads,n_layers):
        super(TypeGAT, self).__init__()

        self.num_e = num_e
        self.num_r = num_r
        self.in_dim = relation_embeddings.shape[1]
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        #self.final_node_embeddings = nn.Parameter(
        #    torch.randn(self.num_nodes, self.out_dim))
        #self.final_entity_embeddings = nn.Parameter(torch.randn(self.num_e, self.out_dim))
        #self.final_relation_embeddings = nn.Parameter(torch.randn(self.num_r, self.out_dim))

        #print(self.node_types.shape)
        #print(self.node_types)

        self.pad = torch.zeros(1, self.out_dim)
        if CUDA:
            self.pad = self.pad.cuda()
        #self.final_node_embeddings = nn.Parameter(torch.cat((torch.randn(self.num_nodes, self.out_dim), self.pad), dim=0))


        #self.entity_embeddings = nn.Parameter(entity_embeddings)
        self.relation_embeddings = nn.Parameter(relation_embeddings)
        self.emb            = RelTemporalEncoding(self.out_dim)
        self.r_layer = nn.Linear(self.out_dim * 2, self.out_dim)

        self.r_score = nn.Linear(self.out_dim, 1)
        self.e_score = nn.Linear(self.out_dim * 2, 1)
        #self.e_layer = nn.Linear(self.out_dim * 3, self.out_dim)
        #self.alpha = nn.Linear(self.out_dim * 2, 1)
        self.norm = nn.LayerNorm(self.out_dim)
        self.max_pooling = nn.AdaptiveAvgPool1d(1)

        self.resnet = nn.Linear(1, 1)

        #self.o_layer = nn.Linear(self.out_dim * 2, self.out_dim)

        self.gat = nn.ModuleList()
        #self.ragg = nn.ModuleList()
        #self.agge = nn.ModuleList()

        self.prconv = GATConv(self.in_dim, self.out_dim)
        #self.rconv = GCNConv(self.in_dim, self.out_dim)
        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, batch_first=True)

        for l in range(self.n_layers):
            self.gat.append(GATConv(self.in_dim, self.out_dim))
            #self.gat.append(RGCNConv(self.in_dim, self.out_dim, num_r, num_bases = 2))
            #self.gat.append(GATConv(self.in_dim, self.out_dim, add_self_loops=False, edge_dim=self.in_dim))

        for conv in self.gat:
            conv.reset_parameters()
        #self.rconv.reset_parameters()
        self.gru.reset_parameters()
        self.r_layer.reset_parameters()



    def forward2(self, path_index, batch_relation, paths, paths_time, lengths,path_r,path_neg_index, batch_his_r):
        #node_inp = self.entity_embeddings
        r_inp = self.relation_embeddings

        # update relations r<-path
        pad_r = torch.cat((r_inp, self.pad), dim=0)
        emb = pad_r[paths]
        emb = self.emb(emb,paths_time)#temporal information
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False).to(paths.device)
        _, hidden = self.gru(packed)
        # out,_ = pad_packed_sequence(out, batch_first=True)
        path_emb = torch.cat((self.pad, hidden.squeeze(0)), dim=0)
        del emb, packed, paths

        pad_r = torch.cat((F.normalize(r_inp, dim=1), pad.to(r_inp.device)), dim=0)
        path_emb = F.normalize(path_emb, dim=1)

        scores = torch.mm(path_emb, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        max_score, max_id = torch.max(scores[path_index], 1)

        scores_r = torch.mm(pad_r, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        his_score = torch.mean(scores_r[batch_his_r], 1)


        #score = max_score+his_score
        score = max_score

        return score, path_emb[path_neg_index], pad_r[path_r]


    def test(self, path_index, batch_relation, paths, lengths, paths_time, batch_his_r):
        r_inp = self.relation_embeddings

        pad = torch.zeros(1, self.out_dim)

        # update relations r<-path
        pad_r = torch.cat((r_inp, pad.to(r_inp.device)), dim=0)
        emb = pad_r[paths]
        emb = self.emb(emb, paths_time)  # temporal information
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        # out,_ = pad_packed_sequence(out, batch_first=True)
        path_emb = torch.cat((self.pad.to(r_inp.device), hidden.squeeze(0)), dim=0)
        
        del emb, packed, paths
        pad_r = torch.cat((F.normalize(r_inp, dim=1), pad.to(r_inp.device)), dim=0)
        #pad_r = F.normalize(pad_r, dim=1)
        path_emb = F.normalize(path_emb, dim=1)


        #print(path_emb[path_index].shape)

        scores = torch.mm(path_emb, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        max_score, max_id = torch.max(scores[path_index], 1)

        scores_r = torch.mm(pad_r, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        #sc = scores_r[batch_his_r]
        #sign_sc = torch.sign(sc).int()
        #his_score = torch.from_numpy(np.count_nonzero(sign_sc.numpy(), axis=1))
        #non_zero_sc = np.count_nonzero(sign_sc.numpy(), axis=1)+0.00001

        his_score = torch.mean(scores_r[batch_his_r], 1)

        score = max_score + his_score
        #score = his_score

        return score, max_score, his_score

    def __repr__(self):
        return self.__class__.__name__


