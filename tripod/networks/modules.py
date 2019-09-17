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
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_size=300, encoder_layers=2, dropout=0.5, nn_type=nn.GRU):
        super(Encoder, self).__init__()

        self.rnn = nn_type(input_size=input_size, hidden_size=encoder_size, bidirectional=True,
                           num_layers=encoder_layers, dropout=dropout)
        self.fc = nn.Linear(encoder_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        outputs, hidden = self.rnn(src)
        if isinstance(hidden, list) or isinstance(hidden, tuple):  # we have a LSTM
            hidden = hidden[1]
        hidden = torch.tanh(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(outputs.permute(1, 0, 2)), hidden


class DecoderMLP(nn.Module):
    def __init__(self, input_size, output_dim, hidden_layers=[500], dropout=0.5, internal_embeddings_size=300):
        super(DecoderMLP, self).__init__()
        inp_size = input_size * 2 + internal_embeddings_size
        hidden_list = []
        for hidden_size in hidden_layers:
            hidden_list.append(nn.Linear(inp_size, hidden_size, bias=True))
            hidden_list.append(nn.Tanh())
            hidden_list.append(nn.Dropout(p=dropout, inplace=True))
            inp_size = hidden_size

        hidden_list.append(nn.Linear(inp_size, output_dim, bias=True))
        self.mlp = nn.Sequential(hidden_list)
        self.embedding = nn.Embedding(output_dim + 1, internal_embeddings_size)
        self.dropout = nn.Dropout(dropout)
        self._PAD_TOKEN = output_dim

    def forward(self, input_tokens, conditioning_vect, partition_dropout=False):
        import copy
        it_list = copy.deepcopy(input_tokens)
        it_list.insert(0, self._PAD_TOKEN)
        it_list.insert(0, self._PAD_TOKEN)
        embedded_1 = self.dropout(self.embedding(it_list[:-2]))
        embedded_2 = self.dropout(self.embedding(it_list[1:-1]))
        embedded = torch.cat((embedded_1, embedded_2), dim=1)
        conditioning_vect = conditioning_vect.unsqueeze(1).repeat(1, input_tokens.shape[1], 1)
        if partition_dropout:
            if self.training:
                mask_x = []
                mask_cond = []
                for _ in range(input_tokens.shape[1]):
                    p1 = random.random()
                    p2 = random.random()
                    m1 = 1
                    m2 = 1
                    if p1 < 0.34:
                        m1 = 0
                        m2 = 2
                    if p2 < 0.34:
                        m1 *= 2
                        m2 = 0
                    mask_x.append(m1)
                    mask_cond.append(m2)
                mask_x = torch.tensor(mask_x, device=conditioning_vect.device.type, dtype=torch.float)
                mask_cond = torch.tensor(mask_cond, device=conditioning_vect.device.type, dtype=torch.float)
                mask_x = mask_x.repeat(embedded.shape[0], 1, 1).permute(0, 2, 1)
                mask_cond = mask_cond.repeat(embedded.shape[0], 1, 1).permute(0, 2, 1)
                mlp_input = torch.cat((mask_x * embedded, mask_cond * conditioning_vect), dim=2)
            else:
                mlp_input = torch.cat((embedded, conditioning_vect), dim=2)
        else:
            mlp_input = torch.cat((embedded, conditioning_vect), dim=2)




class DecoderLSTM(nn.Module):
    def __init__(self, input_size, output_dim, decoder_size=300, decoder_layers=2, dropout=0.5, attention=None,
                 nn_type=nn.GRU, internal_embeddings_size=300):
        super(DecoderLSTM, self).__init__()
        self.attention = attention

        self.rnn = nn_type(input_size + internal_embeddings_size, decoder_size, num_layers=decoder_layers)
        self.out = nn.Linear(decoder_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_dim, internal_embeddings_size)

    def forward(self, input_tokens, conditioning_vect, hidden=None, partition_dropout=False):
        embedded = self.dropout(self.embedding(input_tokens))
        conditioning_vect = conditioning_vect.unsqueeze(1).repeat(1, input_tokens.shape[1], 1)
        if partition_dropout:
            if self.training:
                mask_x = []
                mask_cond = []
                for _ in range(input_tokens.shape[1]):
                    p1 = random.random()
                    p2 = random.random()
                    m1 = 1
                    m2 = 1
                    if p1 < 0.34:
                        m1 = 0
                        m2 = 2
                    if p2 < 0.34:
                        m1 *= 2
                        m2 = 0
                    mask_x.append(m1)
                    mask_cond.append(m2)
                mask_x = torch.tensor(mask_x, device=conditioning_vect.device.type, dtype=torch.float)
                mask_cond = torch.tensor(mask_cond, device=conditioning_vect.device.type, dtype=torch.float)
                mask_x = mask_x.repeat(embedded.shape[0], 1, 1).permute(0, 2, 1)
                mask_cond = mask_cond.repeat(embedded.shape[0], 1, 1).permute(0, 2, 1)
                rnn_input = torch.cat((mask_x * embedded, mask_cond * conditioning_vect), dim=2).permute(1, 0, 2)
            else:
                rnn_input = torch.cat((embedded, conditioning_vect), dim=2).permute(1, 0, 2)
        else:
            rnn_input = torch.cat((embedded, conditioning_vect), dim=2).permute(1, 0, 2)
        if hidden is None:
            output, hidden = self.rnn(rnn_input)
        else:
            output, hidden = self.rnn(rnn_input, hidden)
        output = self.out(output)
        return output.permute(1, 0, 2), hidden


class AttentionGST(nn.Module):
    def __init__(self, gst_dim, conditioning_size, num_gst, output_size):
        super(AttentionGST, self).__init__()

        self.dec_hid_dim = conditioning_size
        self.num_gst = num_gst

        self.gst = nn.Embedding(num_gst, gst_dim)
        self.attn = nn.Linear(gst_dim + conditioning_size, conditioning_size)
        self.v = nn.Parameter(torch.rand(conditioning_size))
        self.out = nn.Linear(gst_dim, output_size)

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        src_len = self.num_gst
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        unfolded_gst = torch.tensor([[i for i in range(self.num_gst)] for _ in range(batch_size)],
                                    device=hidden.device.type, dtype=torch.long)
        encoder_outputs = self.gst(unfolded_gst)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = F.softmax(torch.bmm(v, energy).squeeze(1), dim=1)
        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs).squeeze(1)
        return attention, self.out(weighted)


class AttentionMemory(nn.Module):
    def __init__(self, gst_dim, conditioning_size, num_gst, num_cells, output_size):
        super(AttentionMemory, self).__init__()

        self.dec_hid_dim = conditioning_size
        self.num_gst = num_gst
        self.num_cells = num_cells

        self.gst = nn.ModuleList([nn.Embedding(num_gst, gst_dim) for _ in range(num_cells)])
        self.attn = nn.ModuleList([nn.Linear(gst_dim + conditioning_size, conditioning_size) for _ in range(num_cells)])
        self.v = nn.ParameterList([nn.Parameter(torch.rand(conditioning_size)) for _ in range(num_cells)])
        self.out = nn.Linear(gst_dim, output_size)

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        src_len = self.num_gst
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        unfolded_gst = torch.tensor([[i for i in range(self.num_gst)] for _ in range(batch_size)],
                                    device=hidden.device.type, dtype=torch.long)

        att_list = []
        weighted_list = []
        for ii in range(self.num_cells):
            encoder_outputs = self.gst[ii](unfolded_gst)
            energy = torch.tanh(self.attn[ii](torch.cat((hidden, encoder_outputs), dim=2)))
            energy = energy.permute(0, 2, 1)
            v = self.v[ii].repeat(batch_size, 1).unsqueeze(1)
            attention = F.softmax(torch.bmm(v, energy).squeeze(1), dim=1)
            a = attention.unsqueeze(1)
            weighted = torch.bmm(a, encoder_outputs).squeeze(1)

            att_list.append(attention)
            weighted_list.append(weighted.unsqueeze(1))
        weighted_list = torch.cat(weighted_list, dim=1)
        return torch.cat(att_list, dim=1), self.out(torch.mean(weighted_list, 1, True).squeeze(1))


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, output_size):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        self.out = nn.Linear(enc_hid_dim * 2, output_size)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = F.softmax(torch.bmm(v, energy).squeeze(1), dim=1)

        a = attention.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.squeeze(1)

        return attention, self.out(weighted)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        output = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
