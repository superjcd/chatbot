import torch
import pkuseg
from settings import *
from data_transform import loadPrepareData, indexesFromSentence, trimRareWords
from utilis import normalizeInputsentence, load_model_from_file, GreedySearchDecoder




def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeInputsentence(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ''.join(output_words))

        except KeyError:
            print("我听不懂你说啥")


def bot_answer_api(encoder, decoder, input_sentence, searcher, voc):
    '''
     基于用户对的输入语句， 返回回答
    :param encoder: 编码器
    :param decoder:  解码器
    :param input_sentence:  输入语句
    :param searcher:  贪婪搜素
    :param voc: 字典
    :return:  回答
    '''
    try:
        input_sentence = normalizeInputsentence(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ''.join(output_words)
    except KeyError:
        return None




if __name__ == '__main__':
    voc, pairs = loadPrepareData(corpus_name='qingyun', datafile='corpus/qingyun_seg')
    trimRareWords(voc, pairs, MIN_COUNT)
    encoder, decoder = load_model_from_file(voc, file=EvalFile)
    searcher = GreedySearchDecoder(encoder, decoder)
    evaluateInput(encoder, decoder, searcher, voc)