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

import sys
import json


class Encodings:
    def __init__(self, dataset=None, encoding_type='character', cutoff=5, verbose=False):
        self.token2int = {'<PAD>': 0, '<UNK>': 1, '<START>': 2}
        self.token_list = ['<PAD>', '<UNK>', '<START>']
        self.verbose = verbose
        if dataset is not None:
            self.update_from_dataset(dataset, cutoff=cutoff)

    def update_from_dataset(self, dataset, cutoff=5):
        if self.verbose:
            sys.stdout.write('Updating encodings... ')
            sys.stdout.flush()
        token2count = {}
        if dataset.tokens is not None:
            seqs = [dataset.tokens]
        else:
            seqs = dataset.sequences

        for seq in seqs:
            for token in seq:
                if token not in token2count:
                    token2count[token] = 1
                else:
                    token2count[token] += 1

        for token in token2count:
            if token2count[token] >= cutoff:
                self.token2int[token] = len(self.token2int)
                self.token_list.append(token)
        if self.verbose:
            sys.stdout.write(
                'found {0} unique tokens and pruned to {1}\n'.format(len(token2count), len(self.token2int)))

    def save(self, file_path):
        if self.verbose:
            sys.stdout.write('Storing \'{0}\'... '.format(file_path))
            sys.stdout.flush()
        json.dump(self.token2int, open(file_path, 'w'))
        if self.verbose:
            sys.stdout.write('done\n')
            sys.stdout.flush()

    def load(self, file_path):
        if self.verbose:
            sys.stdout.write('Loading \'{0}\'... '.format(file_path))
            sys.stdout.flush()
        self.token2int = json.load(open(file_path))
        self.token_list = [0 for _ in range(len(self.token2int))]
        for token in self.token2int:
            self.token_list[self.token2int[token]] = token
        if self.verbose:
            sys.stdout.write('loaded {0} tokens\n'.format(len(self.token2int)))
            sys.stdout.flush()


class Dataset:
    def __init__(self, path, type='character'):
        assert (type == 'character' or type == 'token' or type == 'bpe')
        self.sequences = None
        self.tokens = None
        with open(path) as f:
            if type == 'character':
                self.tokens = list(f.read())
            elif type == 'token':
                self.sequences = []
                for line in f.readlines():
                    self.sequences.append(line.split(' '))
            else:
                print("Byte pair encodings is not supported yet")
                sys.exit(1)
            # data = f.read()
            # if type == 'character':
            #    self.tokens = list(data)
            # elif type == 'token':
            #    self.tokens = data.split(' ')
            # else:
            #    print("Byte pair encodings is not supported yet")
            #    sys.exit(1)
