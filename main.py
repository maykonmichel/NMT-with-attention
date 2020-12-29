import io
import re

import tensorflow as tf
import unicodedata
from sklearn.model_selection import train_test_split

path_to_file = "./por-eng/por.txt"


# Converte o arquivo Unicode para ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # criando um espaço entre uma palavra e a pontuação que a segue
    # eg: "he is a boy." => "he is a boy ."
    # Referência:
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adicionar um marcador de início e fim à frase
    # para que o modelo saiba quando começar e parar de prever.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove os acentos
# 2. Limpa as frases
# 3. Retorne pares de palavras no formato: [INGLÊS, PORTUGUÊS]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    target, input, _ = zip(*word_pairs)

    return target, input


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # criando pares de entrada e saída limpos
    target_lang, input_lang = create_dataset(path, num_examples)

    input_tensor, input_lang_tokenizer = tokenize(input_lang)
    target_tensor, target_lang_tokenizer = tokenize(target_lang)

    return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer


num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Criação de conjuntos de treinamento e validação usando uma divisão 80-20
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
    train_test_split(input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # consulta forma de estado oculto == (batch_size, tamanho oculto)
        # query_with_time_axis shape == (batch_size, 1, tamanho oculto)
        # valores forma == (batch_size, max_len, tamanho oculto)
        # estamos fazendo isso para transmitir a adição ao longo do eixo do tempo para calcular a pontuação
        query_with_time_axis = tf.expand_dims(query, 1)

        # forma de pontuação == (batch_size, max_length, 1)
        # obtemos 1 no último eixo porque estamos aplicando pontuação a self.V
        # a forma do tensor antes de aplicar self.V é (batch_size, max_length, unidades)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights forma == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # forma do context_vector após soma == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
