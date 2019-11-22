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
from tripod.networks.modules import Encoder, DecoderLSTM, Attention, AttentionGST, AttentionMemory


class TripodModel(nn.Module):
    def __init__(self, encodings, embeddings_size=300, encoder_size=300, encoder_layers=2, decoder_size=300,
                 decoder_layers=1,
                 num_gst=20, num_cells=50, cell_size=300, dropout=0.5):
        super(TripodModel, self).__init__()
        self.encodings = encodings

        self.encoder_sum = Encoder(embeddings_size, encoder_size * 2, encoder_size=encoder_size, dropout=dropout,
                                   nn_type=nn.LSTM, encoder_layers=encoder_layers)
        self.attn_sum = Attention(encoder_size, encoder_size * 2, embeddings_size)
        self.decoder_sum = DecoderLSTM(embeddings_size, len(encodings.token2int), decoder_size, dropout=dropout,
                                       nn_type=nn.LSTM, decoder_layers=decoder_layers)

        self.encoder_gst = Encoder(embeddings_size, encoder_size * 2, encoder_size=encoder_size, dropout=dropout,
                                   nn_type=nn.LSTM, encoder_layers=encoder_layers)
        self.attn_gst = AttentionGST(embeddings_size, encoder_size * 2, num_gst, embeddings_size)
        self.decoder_gst = DecoderLSTM(embeddings_size, len(encodings.token2int), decoder_size, dropout=dropout,
                                       nn_type=nn.LSTM, decoder_layers=decoder_layers)

        self.encoder_mem = Encoder(embeddings_size, encoder_size * 2, encoder_size=encoder_size, dropout=dropout,
                                   nn_type=nn.LSTM, encoder_layers=encoder_layers)
        self.attn_mem = AttentionMemory(cell_size, encoder_size * 2, num_gst, num_cells, embeddings_size)
        self.decoder_mem = DecoderLSTM(embeddings_size, len(encodings.token2int), decoder_size, dropout=dropout,
                                       nn_type=nn.LSTM, decoder_layers=decoder_layers)
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
        output_sum, hidden_sum = self.encoder_sum(input_emb)
        att_sum, cond_sum = self.attn_sum(hidden_sum, output_sum)
        out_sum, hidden_sum = self.decoder_sum(x, cond_sum, partition_dropout=partition_dropout)

        # GST-based embeddings
        output_gst, hidden_gst = self.encoder_gst(input_emb)
        att_gst, cond_gst = self.attn_gst(hidden_gst)
        out_gst, hidden_gst = self.decoder_gst(x, cond_gst, partition_dropout=partition_dropout)

        # MEM-based embeddings
        output_mem, hidden_mem = self.encoder_mem(input_emb)
        att_mem, cond_mem = self.attn_mem(hidden_mem)
        out_mem, hidden_mem = self.decoder_mem(x, cond_mem, partition_dropout=partition_dropout)
        if not return_attentions:
            return out_sum, out_gst, out_mem
        else:
            return out_sum, out_gst, out_mem, att_sum, att_gst, att_mem, cond_sum, cond_gst, cond_mem

    def compute_repr(self, x):
        input_emb = self.embedding(x)
        # summary-based embeddings
        output_sum, hidden_sum = self.encoder_sum(input_emb)
        att, cond_sum = self.attn_sum(hidden_sum, output_sum)
        # GST-based embeddings
        output_gst, hidden_gst = self.encoder_gst(input_emb)
        att, cond_gst = self.attn_gst(hidden_gst)
        # from ipdb import set_trace
        # set_trace()

        # MEM-based embeddings
        output_mem, hidden_mem = self.encoder_mem(input_emb)
        att, cond_mem = self.attn_mem(hidden_mem)
        return torch.cat((cond_sum, cond_gst, cond_mem), dim=1)

    def generate(self, x):
        input_emb = self.embedding(x)
        # summary-based embeddings
        output_sum, hidden_sum = self.encoder_sum(input_emb)
        att, cond_sum = self.attn_sum(hidden_sum, output_sum)
        hidden_sum = None
        inp = x[0:, 0].unsqueeze(1)
        y_sum_list = []

        for ii in range(x.shape[1]):
            out_sum, hidden_sum = self.decoder_sum(inp, cond_sum, hidden=hidden_sum)
            inp = []
            for tt in range(x.shape[0]):
                inp.append(torch.softmax(out_sum[tt, 0, :], dim=0).multinomial(1).unsqueeze(0))
            inp = torch.cat(inp, dim=0)
            y_sum_list.append(inp)

        y_sum_list = torch.cat(y_sum_list, dim=1)

        # GST-based embeddings
        output_gst, hidden_gst = self.encoder_gst(input_emb)
        att, cond_gst = self.attn_gst(hidden_gst)
        hidden_gst = None
        inp = x[0:, 0].unsqueeze(1)
        y_gst_list = []

        for ii in range(x.shape[1]):
            out_gst, hidden_gst = self.decoder_gst(inp, cond_gst, hidden=hidden_gst)
            inp = []
            for tt in range(x.shape[0]):
                inp.append(torch.softmax(out_gst[tt, 0, :], dim=0).multinomial(1).unsqueeze(0))
            inp = torch.cat(inp, dim=0)
            y_gst_list.append(inp)

        y_gst_list = torch.cat(y_gst_list, dim=1)

        # MEM-based embeddings
        output_mem, hidden_mem = self.encoder_mem(input_emb)
        att, cond_mem = self.attn_mem(hidden_mem)
        hidden_mem = None
        inp = x[0:, 0].unsqueeze(1)
        y_mem_list = []

        for ii in range(x.shape[1]):
            out_mem, hidden_mem = self.decoder_mem(inp, cond_mem, hidden=hidden_mem)
            inp = []
            for tt in range(x.shape[0]):
                inp.append(torch.softmax(out_mem[tt, 0, :], dim=0).multinomial(1).unsqueeze(0))
            inp = torch.cat(inp, dim=0)
            y_mem_list.append(inp)

        y_mem_list = torch.cat(y_mem_list, dim=1)

        return y_sum_list, y_gst_list, y_mem_list

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))