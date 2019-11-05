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


def _split_corpus(params):
    f = open(params.source_file, 'r')
    f_dst = [open("{0}-{1}".format(params.destination_file, index), 'w') for index in range(params.count)]
    index = 0
    for line in tqdm.tqdm(f.readlines()):
        d_file = f_dst[index]
        d_file.write(line)
        index += 1
        index %= params.count
    for file in f_dst:
        file.close()
    f.close()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.set_description('Splits the source file into n almost equal files')
    parser.add_option('--source', action='store', dest='source_file',
                      help='Location for the input file')
    parser.add_option('--destination-prefix', action='store', dest='destination_file',
                      help='Location for the destination file')
    parser.add_option('--count', action='store', type='int', default=10, dest='count',
                      help='How many parts (default=10)')
    (params, _) = parser.parse_args(sys.argv)
    if params.source_file and params.destination_file:
        _split_corpus(params)
    else:
        sys.stdout.write(
            'Specify source, destination and either store or load an encoder. Use --help for more information')
