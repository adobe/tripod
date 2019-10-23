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
import optparse
import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def store_results(params, centroids):
    f = open(params.destination_file, 'w')
    if params.unique:
        for centroid in centroids:
            line = ' '.join(centroid).strip()
            f.write('{0}\n'.format(line))
    f.close()


def dedup_corpus2(params):
    chencherry = SmoothingFunction()
    centroids = {}
    f_src = open(params.source_file, 'r')

    count = 0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vectorizer = TfidfVectorizer()
    vectorized_input = vectorizer.fit_transform(f_src.readlines())
    vectorized_input = vectorized_input
    tsvd = TruncatedSVD(n_components=20)
    vectorized_input = tsvd.fit_transform(vectorized_input)
    pgb = tqdm.tqdm(vectorized_input[:-1], total=vectorized_input.shape[0] - 1)
    i_src = 0
    distances = []
    for line_src in pgb:
        min_dist = None
        min_index = -1
        for i_dst, line_dst in zip(range(i_src + 1, vectorized_input.shape[0]), vectorized_input[i_src + 1:]):
            if i_src != i_dst:
                dist = np.linalg.norm(line_src - line_dst)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_index = i_dst

                count += 1
        i_src += 1
        distances.append([min_index, min_dist])

    allocated = [False for _ in range(vectorized_input.shape[0] - 1)]
    clusters = {}
    for i_src in range(len(distances)):
        # search top node
        top_node = -1
        distance = distances[i_src][1]
        head = distances[i_src][0]
        while distance < params.threshold:
            top_node = head
            if head >= len(distances):
                break
            distance = distances[head][1]
            head = distances[head][0]
        if top_node == -1:
            top_node = i_src

        if top_node in clusters:
            clusters[top_node] += 1
        else:
            clusters[top_node] = 1

    lines = open(params.source_file, 'r').readlines()
    f_out = open(params.destination_file, 'w')
    for cluster in clusters:
        f_out.write(lines[cluster])
    f_out.close()


def dedup_corpus(params):
    chencherry = SmoothingFunction()
    centroids = {}
    f_src = open(params.source_file, 'r')
    pgb = tqdm.tqdm(f_src.readlines())
    count = 0
    for line in pgb:
        count += 1
        parts = line.split(' ')
        if len(centroids) == 0:
            centroids[line] = 1
        else:
            # scores = [[centroid, sentence_bleu([centroid.split(' ')], parts, smoothing_function=chencherry.method1)] for
            #          centroid in centroids]
            scores = []
            for centroid in centroids:
                score = sentence_bleu([centroid.split(' ')], parts, smoothing_function=chencherry.method1)
                scores.append([centroid, score])
                if score >= params.threshold:
                    break
            scores = np.array(scores)
            max_index = np.argmax(scores[:, 1])
            if float(scores[max_index][1]) > params.threshold:
                centroids[scores[max_index, 0]] += 1
            else:
                centroids[line] = 1
        pgb.set_description('CLUSTERS={0} REDUCTION={1}'.format(len(centroids), len(centroids) / count))

    store_results(params, centroids)
    pass


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.set_description('Reduces number of duplicate entries, while preserving original data distribution. '
                           'We use BLEU scoring for comparison')
    parser.add_option('--source', action='store', dest='source_file',
                      help='Location for the input file')
    parser.add_option('--destination', action='store', dest='destination_file',
                      help='Location for the destination file')
    parser.add_option('--threshold', action='store', type='float', default=0.05, dest='threshold',
                      help='Threshold for merging entries (default=0.05) ')
    parser.add_option('--unique', action='store_true', dest='unique',
                      help='Keep only unique entries (data distribution is modified)')

    (params, _) = parser.parse_args(sys.argv)
    if params.source_file and params.destination_file:
        dedup_corpus2(params)
    else:
        sys.stdout.write(
            'Specify source, destination and either store or load an encoder. Use --help for more information')
