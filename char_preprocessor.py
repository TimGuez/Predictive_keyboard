import string
import re
import unidecode
import numpy as np
import tensorflow as tf
from copy import copy
import pickle

ENCODING = 'utf-8'


class CharPreprocessor:

    def __init__(self, sequence_size, data_path, index_rather_than_vector=True):
        self.sequence_size = sequence_size
        self.data_path = data_path
        self.trained = False
        self.dictionary = []
        self.vectors = []

    def train(self):
        full_text = open(self.data_path, "r", encoding=ENCODING).read()
        self.dictionary, self.vectors = self.sentences_to_vect(full_text)
        self.trained = True

    def transform(self, input):
        i = copy(input)
        _, vectors = self.sentences_to_vect(i)
        return vectors

    @staticmethod
    def remove_punctuation(sentences):
        res = []
        for sentence in sentences:
            res.append(re.sub('\d+', '', sentence))
        return res

    @staticmethod
    def remove_special_char(sentences):
        res = []
        for sentence in sentences:
            res.append(re.sub('[^A-Za-z0-9]+', ' ', sentence))
        return res

    def convert_series_into_sequences(self, serie):
        res = []
        if len(serie) >= self.sequence_size:
            for i in range(0, len(serie) - (self.sequence_size-1)):
                res.append(serie[i:i + self.sequence_size])
        return res


    def sentences_to_vect(self, sentences):
        dictionary = self.dictionary
        vectors = []
        if self.trained:
            for letter in sentences:
                index = -1
                if letter in dictionary:
                    index = dictionary.index(letter)
                vectors.append(index)
        else:
            for letter in sentences:
                index = -1
                if letter in dictionary:
                    index = dictionary.index(letter)
                else:
                    dictionary.append(letter)
                    index = len(dictionary)-1
                vectors.append(index)
        return dictionary, vectors

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

# def compute_accuracy_per_word(self):
#     res = np.matrix(np.zeros((len(self.dictionary), len(self.dictionary))), dtype=float)
#     for current_index, current_word in enumerate(self.dictionary):
#         for comparison_index, comparison_word in enumerate(self.dictionary):
#             i = 0
#             go_on = True
#             while go_on and i < min(len(comparison_word), len(current_word)):
#                 if comparison_word[i] == current_word[i]:
#                     i += 1
#                 else:
#                     go_on = False
#             res[current_index, comparison_index] = i / len(comparison_word)
#     self.accuracy_table = tf.convert_to_tensor(res, dtype='float32')

# def key_stoke_saved(self, y_true, y_pred):
#     '''
#     This function tries to implements
#     :param y_true:
#     :param y_pred:
#     :return the means numbers of common letters between words:
#     '''
#     if not(self.index_rather_than_vector):
#         final = tf.keras.backend.dot(tf.keras.backend.dot(y_true, self.accuracy_table), tf.keras.backend.transpose(y_pred))
#         return tf.keras.backend.mean(final)
#     else:
#         #TODO
#         pass
#     return None

# def get_training_data_preprocessed(self):
#     sequences = self.convert_series_into_sequences(self.vectors)
#     if self.index_rather_than_vector:
#         sequences = np.array(sequences)
#         features = np.array(sequences)[:, 0:self.sequence_size - 1]
#         labels = np.array(sequences)[:, self.sequence_size - 1:self.sequence_size]
#     else:
#         sequences = tf.keras.utils.to_categorical(sequences)
#         features = np.array(sequences)[:, 0:self.sequence_size - 1, :].astype('float16')
#         labels = np.array(sequences)[:, self.sequence_size - 1:self.sequence_size, :].astype('float16')
#     return features, labels

