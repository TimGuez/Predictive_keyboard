import tensorflow as tf
import numpy as np


class BatchGenerator(tf.keras.utils.Sequence):
    """
    This class generates batches of data to avoid overloading the memory
    """

    def __init__(self, batch_size, preprocessor, is_validation=False):
        """
        The function is used to create the BatchGenerator
        :param batch_size: integer - size of generated batches
        :param preprocessor: Preprocessor - preprocessor used to prepare data
        :param is_validation: boolean telling the bacth generator which type of data to suumon (train or validation)
        """
        self.preprocessor = preprocessor
        self.is_validation = is_validation
        if self.is_validation:
            self.vectors = preprocessor.vectors[int(np.floor(len(preprocessor.vectors)*0.9)):]
        else:
            self.vectors = preprocessor.vectors[:int(np.floor(len(preprocessor.vectors)*0.9))]
        self.batch_size = batch_size
        self.sequences = []
        # Chunk let the process of convert_series_into_sequences be lighter in memory
        chunk = len(self.vectors)//100  # Don't hesitate to change chunk size to optimise it to your system
        for i in range(99):
            self.sequences = self.sequences + preprocessor.convert_series_into_sequences(self.vectors[i * chunk: (i+1) * chunk])
        np.random.shuffle(self.sequences)

    def __len__(self):
        return (np.ceil(len(self.sequences) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """
        This function is the heart of it all : it is called by training/validation to have data
        :param idx: batch index
        :return: the batch of the selected id
        """
        chosen_seq = self.sequences[idx * self.batch_size: (idx + 1) * self.batch_size]
        cat_sequences = tf.keras.utils.to_categorical(chosen_seq, num_classes=len(self.preprocessor.dictionary))
        features = np.array(cat_sequences)[:, 0:self.preprocessor.sequence_size - 1, :].astype('int8')
        labels = np.array(cat_sequences)[:, self.preprocessor.sequence_size - 1:self.preprocessor.sequence_size, :].astype('int8')
        return features, labels[:, 0, :]
