import re
import unidecode
import numpy as np
from copy import copy
import pickle

ENCODING = 'utf-8'


class Preprocessor:
    """
    This class is used to prepare data to make it usable by a model.
    It implements many usefull functions to our case.
    """

    def __init__(self, sequence_size, data_path, index_rather_than_vector=True):
        self.sequence_size = sequence_size
        self.data_path = data_path
        self.accuracy_table = None
        self.trained = False
        self.index_rather_than_vector = index_rather_than_vector
        self.dictionary = []
        self.bag_of_words = []
        self.vectors = []

    def train(self):
        """
        Trains the model with provided data
        :return: nothing, everything is loaded inside the preprocessor object
        """
        full_text = open(self.data_path, "r", encoding=ENCODING).read().lower()
        unaccented_text = unidecode.unidecode(full_text)
        sentences_list = unaccented_text.split('.')
        sentences_list = Preprocessor.remove_punctuation(sentences_list)
        sentences_list = Preprocessor.remove_special_char(sentences_list)
        self.dictionary, self.bag_of_words, self.vectors = self.sentences_to_vect(sentences_list)
        self.trained = True

    def transform(self, input):
        """
        Transform the input into vectors, according to the trained preprocessing.
        Warining : the preprocessor has to be trained.
        :param input:
        :return:
        """
        i = copy(input)
        unaccented_text = unidecode.unidecode(i.lower())
        sentences_list = unaccented_text.split('.')
        sentences_list = Preprocessor.remove_punctuation(sentences_list)
        sentences_list = Preprocessor.remove_special_char(sentences_list)
        _, _, vectors = self.sentences_to_vect(sentences_list)
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

    def convert_series_into_sequences(self, series):
        res = []
        for serie in series:
            if len(serie) >= self.sequence_size:
                for i in range(0,len(serie) - (self.sequence_size-1)):
                    res.append(serie[i:i + self.sequence_size])
        return res

    def sentences_to_vect(self, sentences):
        """
        This function loops trough sentences to process all words
        It work for trained or untrained models
        :param sentences: sentences to convert
        :return: dictionary, bag of words (usefull for tfidf), vectors
        """
        dictionary = self.dictionary
        vectors = []
        bag_of_words = self.bag_of_words
        if self.trained:
            vector = []
            sentence = sentences[len(sentences) - 1]
            for word in sentence.split():
                if word in dictionary:
                    index = dictionary.index(word)
                else:
                    index = -1
                vector.append(index)
            vectors.append(vector)
        else:
            for sentence in sentences:
                vector = []
                for word in sentence.split():
                    index = -1
                    if word in dictionary:
                        index = dictionary.index(word)
                        bag_of_words[index] += 1
                    else:
                        dictionary.append(word)
                        index = len(dictionary)-1
                        bag_of_words.append(1)
                    vector.append(index)
                vectors.append(vector)
        return dictionary, bag_of_words, vectors

    def load(self, filename):
        """
        load the object to a chosen destination
        :param filename: destination to load preprocessor from
        :return: nothing, but updates object
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """
        saves the object to a chosen destination
        :param filename: destination to save preprocessor to
        :return: nothing
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def tf_idf(self, min_occurence, max_freq):
        """
        Function filtering dictionnary to reduce dimensions
        :param min_occurence: a word won't be kept if appears less than this number
        :param max_freq: a word's frequency should be beneath this threshold or it will be cut
        :return: filtered indexes, but updates all usefull values inside object
        """
        filtered_indexes = {}
        total_count = np.sum(np.array(self.bag_of_words))
        for index, word in reversed(list(enumerate(self.bag_of_words))):
            if bool(word/total_count >= max_freq) | bool(word < min_occurence):
                filtered_indexes[index] = word
                self.dictionary.pop(index)
                for vector in self.vectors:
                    if word in vector:
                        self.vectors.remove(vector)
        self.filtered_indexes = filtered_indexes

# Old function just in case

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

