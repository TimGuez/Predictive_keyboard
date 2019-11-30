from Preprocessor import Preprocessor
import numpy as np
import tensorflow as tf
import unidecode

"""
This script computes Key Stroke Per Character (KSPC)
"""

# Warning : global variables for all script

preprocessor = Preprocessor(0, '/')
preprocessor.load('preprocessor_save/full.pre')
preprocessor.sequence_size=5
model = tf.keras.models.load_model('log5/2/model.h5')

def compute_kspc(input, next_word):
    """
    This function compute key stroke and character produced for a sentence
    :param input: the cleansed sentence
    :param next_word: word predicted by model
    :return:
    """
    vectors = preprocessor.transform(input)
    sequences = vectors
    cat_sequences = tf.keras.utils.to_categorical(sequences, num_classes=len(preprocessor.dictionary))
    features = np.array(cat_sequences)[:, -(preprocessor.sequence_size - 1):, :].astype('float16')
    x = features.astype('float16')
    predictions = model.predict(x)[-1]
    indexes = predictions.argsort()[-3:]
    guessed_words = map(lambda x : preprocessor.dictionary[x], list(indexes))
    if next_word in guessed_words:
        return 1, len(next_word)
    else:
        return len(next_word), len(next_word)


full_text = open(preprocessor.data_path, "r", encoding='utf-8').read().lower()
unaccented_text = unidecode.unidecode(full_text)
sentences_list = unaccented_text.split('\n')
sentences_list = Preprocessor.remove_punctuation(sentences_list)
sentences_list = Preprocessor.remove_special_char(sentences_list)
ks = 0
char = 0
# Looping trough sentences
for sentence in sentences_list:
    if len(sentence.split(" "))>5:
        words = sentence.split(" ")
        words.remove('')
        try:
            passed = words[0] + " " + words[1] + " " + words[2] + " " + words[3]
        except IndexError:
            print('error')
        words = words[4:]
        next_word = ""
        while len(words) > 1:
            next_word = words.pop(0)
            ks1, char1 = compute_kspc(passed, next_word)
            ks += ks1
            char += char1
            passed_split = passed.split(" ")
            passed = passed_split[1] + " " + passed_split[2] + " " + passed_split[3] + " " + next_word
            print("%.2f" % (ks/char))  # printing the actualized KSPC value
