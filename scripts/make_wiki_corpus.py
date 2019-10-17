import os

def get_files(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if 'wiki_' in file:
                files.append(os.path.join(r, file))
    return files



import sys

if len(sys.argv)<2:
     print("Usage: make_wiki_corpus.py <input_folder> <train_file> <dev_file>")
else:
    source=sys.argv[1]
    train_file=sys.argv[2]
    dev_file=sys.argv[3]

    all_files=get_files(source)
    print(len(all_files))
    f_train=open(train_file, 'w')
    f_dev=open(dev_file, 'w')
    import tqdm
    cnt=0
    for file in tqdm.tqdm(all_files):
        lines=open(file).readlines()
        p_lines=[]
        for line in lines:
            if line.strip()!='' and '</doc' not in line and '<doc id' not in line:
                p_lines.append(line)
        for line in p_lines:
            cnt+=1
            if cnt%10==0:
                f_out=f_dev
            else:
                f_out=f_train
            f_out.write(line)
    f_train.close()
    f_dev.close()
