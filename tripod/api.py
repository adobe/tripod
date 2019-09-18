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

import requests
import sys
import os
import torch
import numpy as np

sys.path.append('')
from tripod.model import TripodModel
from bpe.encoder import Encoder as BPEEncoder
from tripod.io_utils.io import Encodings
from pathlib import Path
from zipfile import ZipFile


class Tripod(object):
    def __init__(self, verbose=False, device='cpu'):
        """
        Creates an empty Tripod instance. Before you can use this instance, you must call .load
        :param verbose: Set this flag to ```True``` to trigger verbose printout
        :param device: Set this flag to switch between devices: cpu, cuda:0 etc.
        """
        self._verbose = verbose
        self._loaded = False

        self._model = None
        self._bpe = None
        self._encodings = None
        self._device = device

        self._model_store = os.path.join(str(Path.home()), ".tripod/models")
        if not os.path.exists(self._model_store):
            os.makedirs(self._model_store)

    def load(self, model_name, force_update=False):
        """
        Use this function to automatically download and load a new model. It will automatically check online if a newer version is available.
        :param model_name: identifier for the model to be loaded
        :param force_update: use this flag to trigger forceful model update - useful if you have an unhealthy model store
        :return: True if the process was successful and False if something failed

        """
        try:
            URL_PREFIX = "https://github.com/adobe/tripod/raw/master/data/trained/"
            model_prefix = os.path.join(self._model_store, model_name)
            must_download = force_update or not os.path.exists(model_prefix + '.best') or not os.path.exists(
                model_prefix + '.encodings')
            model_name_suffixes = ['-aa', '-ab', '-ac', '-ad']
            if must_download:
                # download file parts
                for model_name_suffix in model_name_suffixes:
                    url = "{0}{1}.zip{2}".format(URL_PREFIX, model_name, model_name_suffix)
                    print(url)
                    download_target = model_prefix + '.zip' + model_name_suffix
                    self._download_with_progress_bar(url, download_target)
                    sys.stdout.write('\n')

                # concatenate zip
                download_target = model_prefix + '.zip'
                f_out = open(download_target, 'wb')
                for model_name_suffix in model_name_suffixes:
                    download_part = model_prefix + '.zip' + model_name_suffix
                    f_in = open(download_part, 'rb')
                    f_out.write(f_in.read())
                    f_in.close()
                f_out.close()
                zipfile = ZipFile(download_target, "r")
                zipfile.extractall(self._model_store)
                zipfile.close()
                sys.stdout.write("\nModel extracted successfully.")
                sys.stdout.flush()
            if os.path.exists(model_prefix + '.bpe'):
                self._bpe = BPEEncoder.load(model_prefix + '.bpe')

            self._encodings = Encodings()
            self._encodings.load(model_prefix + '.encodings')
            self._model = TripodModel(self._encodings)
            self._model.load(model_prefix + '.best')
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            return True
        except:
            return False

    def _autopad(self, batch):
        max_len = max([len(seq) for seq in batch])
        for seq in batch:
            n = len(seq)
            for ii in range(max_len - n):
                seq.append('<PAD>')
        # only reason to do this is to make it obvious that batch changes in _make_batches
        return batch

    def _make_batches(self, seqs, batch_size=16):
        batches = []
        batch = []
        for seq in seqs:
            if self._bpe is not None:
                seq = self._bpe.tokenize(seq)
            batch.append(seq)
            if len(batch) == batch_size:
                batch = self._autopad(batch)
                batches.append(batch)
                batch = []

        if len(batch) != 0:
            batch = self._autopad(batch)
            batches.append(batch)

        return self._to_tensor(batches, self._encodings, self._device)

    def __call__(self, seqs, encode_decode=False, batch_size=16):
        output_list = []
        with torch.no_grad():
            batches = self._make_batches(seqs, batch_size=batch_size)
            for batch_x in batches:
                if not encode_decode:
                    representation = self._model.compute_repr(batch_x)
                    for vec in representation:
                        output_list.append(np.asarray(vec.cpu().squeeze(0).numpy()))
                else:
                    pred_sum, pred_gst, pred_mem = self._model.generate(batch_x)

                    val_sum = pred_sum.cpu().numpy()
                    val_gst = pred_gst.cpu().numpy()
                    val_mem = pred_mem.cpu().numpy()
                    for seq_id in range(pred_sum.shape[0]):
                        if self._bpe is not None:
                            token_list_sum = [self._encodings.token_list[zz] for zz in val_sum[seq_id] if
                                              zz != self._encodings.token2int['<UNK>']]
                            token_list_gst = [self._encodings.token_list[zz] for zz in val_gst[seq_id] if
                                              zz != self._encodings.token2int['<UNK>']]
                            token_list_mem = [self._encodings.token_list[zz] for zz in val_mem[seq_id] if
                                              zz != self._encodings.token2int['<UNK>']]
                            output_list.append([self._bpe_decode(token_list_sum, self._bpe),
                                                self._bpe_decode(token_list_gst, self._bpe),
                                                self._bpe_decode(token_list_mem, self._bpe)])
                        else:
                            t_sum = ''.join([self._encodings.token_list[val_sum[seq_id][t_id]] for t_id in
                                             range(pred_sum.shape[1])])
                            t_mem = ''.join([self._encodings.token_list[val_mem[seq_id][t_id]] for t_id in
                                             range(pred_sum.shape[1])])
                            t_gst = ''.join([self._encodings.token_list[val_gst[seq_id][t_id]] for t_id in
                                             range(pred_sum.shape[1])])
                            output_list.append([t_sum, t_gst, t_mem])

        return output_list

    @staticmethod
    def _download_with_progress_bar(url, local_filename):
        r = requests.get(url, stream=True)
        total_size = int(r.headers['Content-Length'].strip())
        current_size = 0
        with open(local_filename, 'wb') as f:
            for buf in r.iter_content(4096 * 16):
                if buf:
                    # request_content.append(buf)
                    f.write(buf)
                    current_size += len(buf)
                    done = int(40 * current_size / total_size)
                    sys.stdout.write("\r[%s%s] %3.1f%%, downloading %.2f/%.2f MB ..." % (
                        '=' * done, ' ' * (40 - done), 100 * current_size / total_size, current_size / 1024 / 1024,
                        total_size / 1024 / 1024))
                    sys.stdout.flush()

    @staticmethod
    def _to_tensor(x, encodings, device):
        batches = []
        for batch in x:
            cb = []
            for seq in batch:
                cs = []
                for token in seq:
                    if token in encodings.token2int:
                        cs.append(encodings.token2int[token])
                    else:
                        cs.append(encodings.token2int['<UNK>'])
                cb.append(cs)
            batches.append(cb)
        return torch.tensor(batches, device=device)

    @staticmethod
    def _bpe_decode(tokens, encoder):
        encoded = []
        for token in tokens:
            if token in encoder.word_vocab:
                encoded.append(encoder.word_vocab[token])
            elif token in encoder.bpe_vocab:
                encoded.append(encoder.bpe_vocab[token])
            else:
                encoded.append(encoder.word_vocab[encoder.UNK])
        decoded = str(next(encoder.inverse_transform([encoded])))
        return decoded
