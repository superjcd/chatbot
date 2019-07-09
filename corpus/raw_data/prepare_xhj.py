###     原始语料形式      ###
#  E
#  M  到碗里来吃翔         #
#  M  你不能找个大一点的碗吗 #
###########################
import pkuseg
import re

# 实例化分词工具
seg = pkuseg.pkuseg()

# 脏数据列表
noise_word = ['M', '\n']


def sentence_cleaner(sentence):
    '''
    清除sentence
    :param sentence:
    :return:
    '''
    for noise in noise_word:
        sentence = re.sub(noise, '', sentence)
    return sentence


def load_conversation_pairs(file_loc):
    # 读取所有文本到list形式
    datas = []
    for data in open(file_loc, 'r'):
        if not data.startswith('E'): # 去掉以E开头的句子
            datas.append(data)
    # 获取问答对
    pairs = []
    pair = []
    for i, sentence in enumerate(datas, 1):
        # 清理sentence
        clean_sentence = sentence_cleaner(sentence)
        pair.append(clean_sentence)
        if i % 2 == 0:
            #print(pair)
            pairs.append(pair)
            pair = []
    return pairs



if __name__ == '__main__':
    pairs = load_conversation_pairs('xhj_seg')
    print(pairs)


