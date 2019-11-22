#
# Author: Tiberiu Boros
#
# Copyright (c) 2019 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import random
from tripod.networks.modules import Encoder, DecoderLSTM, Attention, AttentionGST, AttentionMemory


class TripodModel2(nn.Module):
    def __init__(self, encodings, embeddings_size=300, encoder_size=300, encoder_layers=2, decoder_size=300,
                 decoder_layers=1,
                 num_gst=20, num_cells=50, cell_size=300, dropout=0.5):
        super(TripodModel2, self).__init__()
        self.encodings = encodings

        self.encoder = Encoder(embeddings_size, encoder_size * 2, encoder_size=encoder_size, dropout=dropout,
                               nn_type=nn.LSTM, encoder_layers=encoder_layers)
        self.decoder = DecoderLSTM(embeddings_size * 3, len(encodings.token2int), decoder_size, dropout=dropout,
                                   nn_type=nn.LSTM, decoder_layers=decoder_layers)

        self.attn_sum = Attention(encoder_size, encoder_size * 2, embeddings_size)
        self.attn_gst = AttentionGST(embeddings_size, encoder_size * 2, num_gst, embeddings_size)
        self.attn_mem = AttentionMemory(cell_size, encoder_size * 2, num_gst, num_cells, embeddings_size)

        self.embedding = nn.Embedding(len(encodings.token2int), embeddings_size)

        # LSTM initialization
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and "rnn" in name:  # forget bias
                nn.init.zeros_(param.data)
                param.data[param.size()[0] // 4:param.size()[0] // 2] = 1

    def forward(self, x, return_attentions=False, partition_dropout=False):
        input_emb = self.embedding(x)
        # summary-based embeddings
        output, hidden = self.encoder(input_emb)
        att_sum, cond_sum = self.attn_sum(hidden, output)
        att_gst, cond_gst = self.attn_gst(hidden)
        att_mem, cond_mem = self.attn_mem(hidden)
        cond_sum_ext = cond_sum.unsqueeze(1).repeat(1, x.shape[1], 1)
        cond_gst_ext = cond_gst.unsqueeze(1).repeat(1, x.shape[1], 1)
        cond_mem_ext = cond_mem.unsqueeze(1).repeat(1, x.shape[1], 1)

        cond = torch.relu(torch.cat([cond_sum_ext, cond_gst_ext, cond_mem_ext],
                                    dim=2))  # cond_sum_ext + cond_gst_ext + cond_mem_ext
        if self.training:
            cond = torch.dropout(cond, 0.5, True)

        out_sum, hidden_sum = self.decoder(x, cond, partition_dropout=partition_dropout)

        if not return_attentions:
            return out_sum
        else:
            return out_sum, att_sum, att_gst, att_mem, cond_sum, cond_gst, cond_mem

    def compute_repr(self, x):
        input_emb = self.embedding(x)
        output, hidden = self.encoder(input_emb)
        att_sum, cond_sum = self.attn_sum(hidden, output)
        att_gst, cond_gst = self.attn_gst(hidden)
        att_mem, cond_mem = self.attn_mem(hidden)
        return torch.cat((cond_sum, cond_gst, cond_mem), dim=1)

    def generate(self, x):
        input_emb = self.embedding(x)
        output, hidden = self.encoder(input_emb)
        att_sum, cond_sum = self.attn_sum(hidden, output)
        att_gst, cond_gst = self.attn_gst(hidden)
        att_mem, cond_mem = self.attn_mem(hidden)
        cond_sum_ext = cond_sum.unsqueeze(1)#.re, 1)
        cond_gst_ext = cond_gst.unsqueeze(1)#.repeat(1, x.shape[1], 1)
        cond_mem_ext = cond_mem.unsqueeze(1)#.repeat(1, x.shape[1], 1)

        cond = torch.relu(torch.cat([cond_sum_ext, cond_gst_ext, cond_mem_ext],
                                    dim=2))
        y_list = []
        hidden = None
        inp = x[0:, 0].unsqueeze(1)
        for ii in range(x.shape[1]):
            out, hidden = self.decoder(inp, cond, hidden=hidden)
            inp = []
            for tt in range(x.shape[0]):
                inp.append(torch.softmax(out[tt, 0, :], dim=0).multinomial(1).unsqueeze(0))
            inp = torch.cat(inp, dim=0)
            y_list.append(inp)

        y_list = torch.cat(y_list, dim=1)

        return y_list

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
