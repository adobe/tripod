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
import torch.nn as nn
import torch
import tqdm
import numpy as np

sys.path.append('')

from tripod.networks.tripod2 import TripodModel2


def _eval(model, batches, criterion, encodings):
    model.eval()
    total_loss = 0
    total_sum = 0
    with torch.no_grad():
        pgb = tqdm.tqdm(batches, ncols=80, desc='\tEvaluating loss=NaN')
        cnt = 0
        for batch in pgb:
            cnt += 1
            batch_x = []
            batch_y = []
            for x, y in batch:
                batch_x.append(x)
                batch_y.append(y)
            batch_x = _to_tensor(batch_x, encodings, params.device)
            batch_y = _to_tensor(batch_y, encodings, params.device)

            pred_sum = model(batch_x)
            loss_sum = criterion(pred_sum.contiguous().view(pred_sum.shape[0] * pred_sum.shape[1], pred_sum.shape[2]),
                                 batch_y.view(-1))

            loss_tot = loss_sum
            total_loss += loss_tot.item()
            total_sum += loss_sum.item()

            pgb.set_description(desc='\tEvaluating loss={0:.5f}   '.format(total_loss / cnt))
    return total_sum / cnt


def _get_batches(dataset, params):
    batches = []
    if dataset.tokens is not None:
        num_sequences = len(dataset.tokens) // params.sequence_size
        if len(dataset.tokens) % params.sequence_size != 0:
            num_sequences += 1
    else:
        num_sequences = len(dataset.sequences)

    cb = []
    for seq_id in range(num_sequences):
        if dataset.tokens is not None:
            start = seq_id * params.sequence_size
            stop = min(len(dataset.tokens), start + params.sequence_size)

            x = dataset.tokens[start:stop - 1]
            y = dataset.tokens[start:stop]
        else:
            x = dataset.sequences[seq_id][:-1]
            y = dataset.sequences[seq_id]
            if len(x) >= params.sequence_size:
                x = x[:params.sequence_size - 1]
                y = y[:params.sequence_size]

        x.insert(0, '<START>')

        for _ in range(params.sequence_size - len(x)):
            x.append('<PAD>')
            y.append('<PAD>')
        cb.append([x, y])
        if len(cb) == params.batch_size:
            batches.append(cb)
            cb = []
    if len(cb) != 0:
        batches.append(cb)

    return batches


def _to_tensor(x, encodings, device):
    x_int = []
    for seq in x:
        cs = []
        for token in seq:
            if token in encodings.token2int:
                cs.append(encodings.token2int[token])
            else:
                cs.append(encodings.token2int['<UNK>'])
        x_int.append(cs)
    return torch.tensor(x_int, device=device)


def generate_repr(params):
    import numpy as np
    from tripod.io_utils.io import Encodings
    encodings = Encodings()
    encodings.load(params.output + '.encodings')
    bpe_encoder = None
    if params.bpe_encoder is not None:
        from bpe import Encoder as BPEEncoder
        bpe_encoder = BPEEncoder.load(params.bpe_encoder)
    model = TripodModel2(encodings)
    model.load(params.output + '.bestGST')
    model.to(params.device)
    model.eval()

    with torch.no_grad():
        with open(params.input_file) as f:
            data = f.read()
            if bpe_encoder is not None:
                data = bpe_encoder.tokenize(data)
            batch_x = [data]
            batch_x = _to_tensor(batch_x, encodings, params.device)
            representation = model.compute_repr(batch_x)
            sys.stdout.write(str(np.asarray(representation.cpu().numpy())) + '\n')


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


def run_tripod(params):
    from tripod.io_utils.io import Dataset
    from tripod.io_utils.io import Encodings
    dataset = Dataset(params.input_file)
    encodings = Encodings()
    encodings.load(params.output + '.encodings')
    model = TripodModel2(encodings)
    model.load(params.output + '.bestGST')
    model.to(params.device)
    model.eval()
    bpe_encoder = None
    if params.bpe_encoder is not None:
        dataset.sequences = []
        dataset.tokens = None
        from bpe import Encoder as BPEEncoder
        bpe_encoder = BPEEncoder.load(params.bpe_encoder)
        for line in open(params.input_file).readlines():
            dataset.sequences.append(bpe_encoder.tokenize(line))

    batches = _get_batches(dataset, params)
    token_list = ''
    with torch.no_grad():
        for batch in batches:
            for seq in batch:

                batch_x = []
                for x in seq[0]:
                    batch_x.append(x)
                tmp = batch_x[1:]

                for ii in range(len(tmp)):
                    if tmp[ii] == '<PAD>':
                        tmp = tmp[:ii]
                        break
                if bpe_encoder is not None:
                    orig = _bpe_decode(tmp, bpe_encoder)
                else:
                    orig = tmp
                batch_x = _to_tensor([batch_x], encodings, params.device)

                pred_sum = model.generate(batch_x)

                val_sum = pred_sum.cpu().numpy()

                for seq_id in range(pred_sum.shape[0]):
                    if bpe_encoder is not None:
                        token_list_sum = [encodings.token_list[zz] for zz in val_sum[seq_id] if
                                          zz != encodings.token2int['<UNK>']]
                        sys.stdout.write('ORIG: ' + orig + '\n\n')
                        sys.stdout.write('SUM: ' + _bpe_decode(token_list_sum, bpe_encoder) + '\n\n')
                        token_list = token_list_sum
                        sys.stdout.write('=' * 20)
                        sys.stdout.write('\n\n\n')
                    else:
                        for t_id in range(pred_sum.shape[1]):
                            token_list += encodings.token_list[val_sum[seq_id][t_id]]
                            sys.stdout.write(encodings.token_list[val_sum[seq_id][t_id]])
                            sys.stdout.flush()

                        sys.stdout.write('\n')

    with open(params.output_file, 'w') as f:
        f.write(_bpe_decode(token_list, bpe_encoder) + '\n')
        f.close()


def train_tripod(params):
    import joblib
    from tripod.io_utils.io import Dataset
    from tripod.io_utils.io import Encodings
    from torch.nn import CrossEntropyLoss, NLLLoss
    from torch.optim import Adam
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    sys.stdout.write('Loading {0}...'.format(params.train_file))
    trainset = Dataset(params.train_file, type=params.input_type)
    sys.stdout.write(' found {0} examples\n'.format(len(trainset.sequences)))
    sys.stdout.write('Loading {0}...'.format(params.dev_file))
    devset = Dataset(params.dev_file, type=params.input_type)
    sys.stdout.write(' found {0} examples\n'.format(len(devset.sequences)))
    encodings = Encodings(trainset, verbose=True, cutoff=params.cutoff)
    encodings.save(params.output + '.encodings')
    cosine_sim = nn.CosineSimilarity(dim=0)

    sys.stdout.write('Begining smart initialization of model...\n')
    sys.stdout.write('\tBuilding TF-IDF vectorizer...')
    sys.stdout.flush()

    corpus = open(params.train_file).readlines()
    corpus = [line.strip() for line in corpus]
    vectorizer = TfidfVectorizer()
    vectorized_input = vectorizer.fit_transform(corpus)
    sys.stdout.write('done\n')
    tsvd = TruncatedSVD(n_components=20)
    reduced_data = tsvd.fit_transform(vectorized_input)
    joblib.dump(tsvd, params.output + '.svd')

    joblib.dump(vectorizer, params.output + '.tfidf')
    sys.stdout.write('\tExecuting K-Means clustering...')
    sys.stdout.flush()

    kmeans = KMeans(n_clusters=params.num_gst, n_jobs=-1, max_iter=10000)
    classified_input = kmeans.fit_predict(reduced_data)
    sys.stdout.write('done\n')
    joblib.dump(classified_input, params.output + '.kmeans')
    elements_in_cluster = [0 for _ in range(params.num_gst)]
    for class_index in classified_input:
        elements_in_cluster[class_index] += 1
    for index, count in zip(range(len(elements_in_cluster)), elements_in_cluster):
        sys.stdout.write('\t\tcluster {0} has {1} datapoints\n'.format(index, count))

    sys.stdout.flush()

    model = TripodModel2(encodings, num_gst=params.num_gst)
    model.to(params.device)

    patience_left = params.patience
    epoch = 0

    train_batches = _get_batches(trainset, params)
    dev_batches = _get_batches(devset, params)
    criterion = CrossEntropyLoss(ignore_index=encodings.token2int['<PAD>'])
    criterion_nll = NLLLoss()
    optimizer = Adam(model.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
    #best_sum = #_eval(model, dev_batches, criterion, encodings)
    best_sum = 9999
    sys.stdout.write(
        '\n\tDevset evaluation: summary_loss={0}\n'.format(best_sum))
    sys.stdout.flush()
    while patience_left > 0:
        model.train()
        patience_left -= 1
        epoch += 1
        sys.stdout.write('\nStarting epoch {0}\n'.format(epoch))
        sys.stdout.flush()
        total_loss = 0
        pgb = tqdm.tqdm(train_batches, ncols=80, desc='\tTraining loss=NaN')
        cnt = 0
        for batch in pgb:
            cnt += 1
            batch_x = []
            batch_y = []
            for x, y in batch:
                batch_x.append(x)
                batch_y.append(y)

            detok_x = [' '.join([token for token in line]) for line in batch_x]
            vectorized_x = vectorizer.transform(detok_x)
            reduced_x = tsvd.transform(vectorized_x)
            class_x = kmeans.predict(reduced_x)
            batch_x = _to_tensor(batch_x, encodings, params.device)
            batch_y = _to_tensor(batch_y, encodings, params.device)

            pred_sum, _, att_gst, att_mem, cond_sum, cond_gst, cond_mem = model(batch_x,
                                                                                return_attentions=True,
                                                                                partition_dropout=params.partition_dropout)
            loss_sum = criterion(pred_sum.contiguous().view(pred_sum.shape[0] * pred_sum.shape[1], pred_sum.shape[2]),
                                 batch_y.view(-1))

            loss_cosine_list = []
            for ii in range(pred_sum.shape[0]):
                for jj in range(ii + 1, pred_sum.shape[0]):
                    cosine_1 = cosine_sim(cond_sum[ii], cond_sum[jj])
                    cosine_2 = cosine_sim(cond_gst[ii], cond_gst[jj])
                    cosine_3 = cosine_sim(cond_mem[ii], cond_mem[jj])
                    target = 1.0
                    if class_x[ii] != class_x[jj]:
                        target = -1.0

                    cosine_loss = (torch.abs(target - cosine_1) + torch.abs(target - cosine_2) + torch.abs(
                        target - cosine_3)) / 3
                    loss_cosine_list.append(cosine_loss)
            loss_cosine = sum(loss_cosine_list) / len(loss_cosine_list)
            eps = 1e-8
            loss_att_gst = sum(
                -torch.log(torch.clamp(a_gst[c_x], min=eps)) for a_gst, c_x in zip(att_gst, class_x)) / len(class_x)
            loss_att_mem = sum(
                -torch.log(torch.clamp(a_mem[c_x], min=eps)) for a_mem, c_x in zip(att_mem, class_x)) / len(class_x)

            loss_tot = loss_sum + (loss_att_gst + loss_att_mem) * 0.2 + loss_cosine * 0.6
            total_loss += loss_tot.item()
            pgb.set_description(desc='\tTraining loss={0:.5f}   '.format(total_loss / cnt))
            optimizer.zero_grad()
            loss_tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        sys.stdout.write('\n\tTraiset loss is {0:.5f}\n'.format(total_loss / (len(train_batches))))
        sys.stdout.flush()
        lss_sum = _eval(model, dev_batches, criterion, encodings)
        sys.stdout.write(
            '\n\tDevset evaluation: loss={0}\n'.format(lss_sum))
        sys.stdout.flush()
        if best_sum is None or lss_sum < best_sum:
            best_sum = lss_sum
            patience_left = params.patience
            path = params.output + '.best'
            sys.stdout.write('\tStoring {0}\n'.format(path))
            sys.stdout.flush()
            model.save(path)

        path = params.output + '.last'
        sys.stdout.write('\tStoring {0}\n'.format(path))
        model.save(path)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='To train a model use the following command:\n\t'
                                         'model.py --train=<TRAINING FILE> --dev=<VALIDATION FILE> '
                                         '--output=<OUTPUT PREFIX>\n\n'
                                         'Check "scripts/" folder for options on creating your dataset')
    parser.add_option('--train', action='store', dest='train_file',
                      help='Location of the training file')
    parser.add_option('--dev', action='store', dest='dev_file',
                      help='Location of the validation file')
    parser.add_option('--output', action='store', dest='output', help='Prefix to for output files')
    parser.add_option('--patience', action='store', dest='patience', type='int', default=20,
                      help='Training patience (default 20)')
    parser.add_option('--cutoff', action='store', dest='cutoff', type='int', default=5,
                      help='Token cutoff for encodings (default=5)')
    parser.add_option('--batch-size', action='store', dest='batch_size', type='int', default=32,
                      help='Number of examples in a single batch (default=32)')
    parser.add_option('--sequence-size', action='store', dest='sequence_size', type='int', default=1000,
                      help='Sequence length (default=1000). Use -1 for automatic detection.')
    parser.add_option('--device', action='store', dest='device', default='cpu',
                      help='What device to use (cpu, cuda:0, cuda:1 ...)')
    parser.add_option('--input', action='store', dest='input_file',
                      help='Location of the input file')
    parser.add_option('--model', action='store', dest='output',
                      help='Prefix of the model')
    parser.add_option('--output-file', action='store', dest='output_file',
                      help='Where to store generation results')
    parser.add_option('--generate', action='store_true', dest='generate',
                      help='Generate using learned autoencoder')
    parser.add_option('--compute', action='store_true', dest='compute',
                      help='Compute the representation for an input file')
    parser.add_option('--input-type', action='store', dest='input_type', default='token',
                      help='Input type: character, token, bpe (default=token)',
                      choices=['character', 'token'])
    parser.add_option('--bpe-encoder', action='store', dest='bpe_encoder',
                      help='Location for the BPE encoder')
    parser.add_option('--num-gst', action='store', type='int', default=20, dest='num_gst',
                      help='Number of GST tokens (default=20)')
    parser.add_option('--lr', action='store', dest='lr', help='Learning rate for Adam (default=1e-4)', default=1e-4,
                      type='float')
    parser.add_option('--beta1', action='store', dest='beta1', help='Beta-1 for Adam (default=0.9)', default=0.9,
                      type='float')
    parser.add_option('--beta2', action='store', dest='beta2', help='Beta-2 for Adam (default=0.999)', default=0.999,
                      type='float')
    parser.add_option('--partition-dropout', action='store_true', dest='partition_dropout',
                      help='Do partition dropout when training')

    parser.set_description('Tripod v1.0')

    (params, _) = parser.parse_args(sys.argv)

    if params.train_file and params.dev_file and params.output and params.patience:
        train_tripod(params)
    elif params.generate:
        run_tripod(params)
    elif params.compute:
        generate_repr(params)
    else:
        parser.print_help()
