import sys

if len(sys.argv) < 2:
    print("Usage: split_corpus.py <source> <train> <dev>")
    sys.exit(0)

f_train = open(sys.argv[2], 'w')
f_dev = open(sys.argv[3], 'w')
cnt = 0
for line in open(sys.argv[1]):
    cnt += 1
    if cnt % 10 == 0:
        f_dev.write(line.strip() + '\n')
    else:
        f_train.write(line.strip() + '\n')
f_train.close()
f_dev.close()
