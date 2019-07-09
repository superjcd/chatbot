from vocabulary import Voc
from corpus.raw_data.prepare_xhj import load_conversation_pairs
from corpus.raw_data.prepare_qy import load_conversation_pairs_qy
from settings import MAX_LENGTH, MIN_COUNT, read_voc_method



def readVocs(datafile, corpus_name, method=read_voc_method):
    '''
     读取语料， 加入到字典中
    :param datafile: 分词后的文本位置
    :param corpus_name: 语料名称
    :return:
    '''
    print("开始读取语句...")
    if method == 'qingyun':
        pairs = load_conversation_pairs_qy(datafile)
    elif method == 'xhj':
        pairs = load_conversation_pairs(datafile)
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(p):
    '''
    判断问答对的问题和答案的（分词词数）是不是都小于MAX_LENGTH
    :param p: 问答对
    :return: 布尔值
    '''
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def loadPrepareData(corpus_name, datafile, corpus=None, save_dir=None):
    '''

    :param corpus:
    :param corpus_name: 语料名
    :param datafile: 语料位置
    :param save_dir:
    :return:
    '''
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs



def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



if __name__ == '__main__':
    voc, pairs = loadPrepareData(corpus_name='qingyun',datafile='corpus/qingyun_seg')
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    print(pairs[:10])
    print(voc.word2index['不是'])



