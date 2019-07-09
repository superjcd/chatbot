import pkuseg
from tqdm import tqdm

seg = pkuseg.pkuseg()


file = open('qingyun.csv', 'r')

with open('qingyun_seg', 'w') as f:
    for sentence in tqdm(file):
        seg_sentence = ' '.join(seg.cut(sentence))
        f.write(seg_sentence + '\n')


file.close()