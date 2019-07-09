PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3

MAX_LENGTH = 10  # 最大句子长度
MIN_COUNT = 3    # 最小词数（低于这个将被）


# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 100 # 如果电脑hold的住， 可以大点


## train params
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
checkpoint_iter = 4000
save_dir = 'models'



# 需要调整的参数（重要）
device = 'cpu'
corpus_name = 'qingyun'
data_file = 'corpus/qingyun_seg'
read_voc_method = 'qingyun'

loadFilename = None
EvalFile = '/Users/jiangchaodi/PycharmProjects/NLP/chatbot/models/cb_model/qingyun/2-2_500/4000_checkpoint.tar'
