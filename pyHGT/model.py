import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pack_sequence
import numpy as np
import torch.nn.functional as F
import math
CUDA = torch.cuda.is_available()

class RelTemporalEncoding(nn.Module):

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
    def __init__(self, num_e, num_r, relation_embeddings, out_dim):
        super(TypeGAT, self).__init__()

        self.num_e = num_e
        self.num_r = num_r
        self.in_dim = relation_embeddings.shape[1]
        self.out_dim = out_dim

        self.pad = torch.zeros(1, self.out_dim)
        if CUDA:
            self.pad = self.pad.cuda()

        self.relation_embeddings = nn.Parameter(relation_embeddings)
        self.emb            = RelTemporalEncoding(self.out_dim)

        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, batch_first=True)

        self.gru.reset_parameters()



    def forward2(self, path_index, batch_relation, paths, paths_time, lengths,path_r,path_neg_index, batch_his_r):
        r_inp = self.relation_embeddings

        # update relations r<-path
        pad_r = torch.cat((r_inp, self.pad), dim=0)
        emb = pad_r[paths]
        emb = self.emb(emb,paths_time)#temporal information
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False).to(paths.device)
        _, hidden = self.gru(packed)
        path_emb = torch.cat((self.pad, hidden.squeeze(0)), dim=0)
        del emb, packed, paths

        pad_r = torch.cat((F.normalize(r_inp, dim=1), self.pad.to(r_inp.device)), dim=0)
        #pad_r = F.normalize(pad_r, dim=1)
        path_emb = F.normalize(path_emb, dim=1)

        scores = torch.mm(path_emb, pad_r[batch_relation].t()).t()  # batch*num_paths
        mask = torch.zeros((scores.size(0), scores.size(1))).cuda()
        m_index = min(path_index.size(1), mask.size(1))
        mask = mask.scatter(1, path_index[:, 0:m_index], 1)
        max_score, max_id = torch.max(scores * mask, 1)

        scores_r = torch.mm(pad_r, pad_r.t())[batch_relation]
        his_score = torch.mean(torch.diagonal(scores_r[:, batch_his_r], dim1=0, dim2=1).t(), 1)

        #scores = torch.mm(path_emb, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        #max_score, max_id = torch.max(scores[path_index], 1)

        #scores_r = torch.mm(pad_r, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        #his_score = torch.mean(scores_r[batch_his_r], 1)


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
        path_emb = torch.cat((self.pad.to(r_inp.device), hidden.squeeze(0)), dim=0)
        
        del emb, packed, paths
        pad_r = torch.cat((F.normalize(r_inp, dim=1), pad.to(r_inp.device)), dim=0)
        path_emb = F.normalize(path_emb, dim=1)


        scores = torch.mm(path_emb, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        max_score, max_id = torch.max(scores[path_index], 1)

        scores_r = torch.mm(pad_r, pad_r[batch_relation[0]].unsqueeze(1)).squeeze(1)
        his_score = torch.mean(scores_r[batch_his_r], 1)

        score = max_score + his_score
        #score = his_score

        return score

    def __repr__(self):
        return self.__class__.__name__


