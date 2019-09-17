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

import optparse
import sys
import tqdm
from bpe import Encoder

sys.path.append('')


def run_bpe(params):
    bpe_encoder = Encoder(vocab_size=params.vocab_size, pct_bpe=params.pct_bpe, silent=not params.verbose)
    if params.encoder_load_file:
        sys.stdout.write('Using pre-computed BPE encoder\n')
        sys.stdout.flush()
        bpe_encoder = Encoder.load(params.encoder_load_file)
    else:
        sys.stdout.write('Generating new BPE encoder\n')
        sys.stdout.flush()
        text = open(params.source_file).read().split('\n')
        bpe_encoder.fit(text)
        bpe_encoder.save(params.encoder_save_file)
    f_src = open(params.source_file)
    f_dst = open(params.destination_file, 'w')

    for line in tqdm.tqdm(f_src.readlines()):
        line = line.strip()
        tokens = bpe_encoder.tokenize(line)
        encoded_line = ' '.join(tokens).strip()
        if encoded_line.strip()!='':
            f_dst.write(encoded_line + '\n')
    f_src.close()
    f_dst.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--source', action='store', dest='source_file',
                      help='Location for the input file')
    parser.add_option('--destination', action='store', dest='destination_file',
                      help='Location for the destination file')
    parser.add_option('--store', action='store', dest='encoder_save_file', help='Location where to store the encoder')
    parser.add_option('--load', action='store', dest='encoder_load_file',
                      help='Location where to load the encoder from')
    parser.add_option('--vocab-size', action='store', dest='vocab_size', default=2000, type='int',
                      help='Vocabulary size (default=2000)')
    parser.add_option('--pct-bpe', action='store', dest='pct_bpe', default=0.2, type='float',
                      help='bpe pct value (default=0.2)')
    parser.add_option('--verbose', action='store_true', dest='verbose',
                      help='Be verbose')
    (params, _) = parser.parse_args(sys.argv)
    if params.source_file and params.destination_file and (params.encoder_save_file or params.encoder_load_file):
        run_bpe(params)
    else:
        sys.stdout.write(
            'Specify source, destination and either store or load an encoder. Use --help for more information')
