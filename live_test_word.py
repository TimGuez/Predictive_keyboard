from Preprocessor import Preprocessor
from char_preprocessor import CharPreprocessor
import numpy as np
import tensorflow as tf

""" This script let you test a model with a sentence of your own making"""

# Loading our models -> modify the path to models to test other thing

preprocessor = Preprocessor(0, '/')
preprocessor.load('preprocessor_save/full.pre')
preprocessor.sequence_size=5
model = tf.keras.models.load_model('log5/2/model.h5')
i = input('test :')
# Predicting the next 5 words
for k in range(5):
    vectors = preprocessor.transform(i)
    sequences = preprocessor.convert_series_into_sequences(vectors)
    cat_sequences = tf.keras.utils.to_categorical(sequences, num_classes=len(preprocessor.dictionary))
    features = np.array(cat_sequences)[:, -(preprocessor.sequence_size-1):, :].astype('float16')

    x_train = features.astype('float16')

    predictions = model.predict(x_train)[-1]
    indexes = predictions.argsort()[-1]
    values = np.sort(predictions)[-1]

    print('Next word would be : ' + preprocessor.dictionary[indexes] + ' with probability : ' + str(values))
    i = i + " " + preprocessor.dictionary[indexes]
