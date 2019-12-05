from Preprocessor import Preprocessor
from char_preprocessor import CharPreprocessor
import numpy as np
import tensorflow as tf


preprocessor = CharPreprocessor(0, '/')
preprocessor.load('preprocessor_save/full_char.pre')
preprocessor.sequence_size=40
i = input('test :')
for k in range(10):
    vectors = preprocessor.transform(i)
    sequences = preprocessor.convert_series_into_sequences(vectors)
    cat_sequences = tf.keras.utils.to_categorical(sequences, num_classes=len(preprocessor.dictionary))
    features = np.array(cat_sequences)[:, -(preprocessor.sequence_size-1):, :].astype('float16')

    x_train = features.astype('float16')
    model = tf.keras.models.load_model('final_char/0/model.h5')
    predictions = model.predict(x_train)[-1]
    indexes = predictions.argsort()[-1]
    values = np.sort(predictions)[-1]

    print('Next word would be : ' + preprocessor.dictionary[indexes] +' with probability : ' + str(values))
    i = i + " " + preprocessor.dictionary[indexes]