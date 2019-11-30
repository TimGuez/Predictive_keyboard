from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random
import shutil

from absl import app
from absl import flags
import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorboard.plugins.hparams import api as hp
from Preprocessor import Preprocessor
from BatchGenerator import BatchGenerator

# CONSTANTS

RATIO_TEST_TRAIN = 0.8

# FLAGS

flags.DEFINE_integer(
    "num_session_groups",
    1,
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "logdir",
    "log7",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    100,
    "Summaries will be written every n steps, where n is the value of "
    "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    4,
    "Number of epochs per trial.",
)

assert tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)  # just want to be sure my GPU is selected


# HYPER PARAMETERS

HP_SEQUENCE_SIZE = hp.HParam('sequence_size', hp.IntInterval(10, 40))
HP_HIDDEN_LAYERS = hp.HParam('hidden_layers', hp.IntInterval(15, 250))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.1, 0.5))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(10, 20))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0001, 0.65))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adagrad"]))


HPARAMS = [HP_SEQUENCE_SIZE, HP_HIDDEN_LAYERS, HP_LEARNING_RATE, HP_OPTIMIZER, HP_BATCH_SIZE, HP_DROPOUT]



# METRICS

# Nothing exotic here
METRICS = [
    hp.Metric(
        "batch_categorical_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "epoch_categorical_accuracy",
        group="validation",
        display_name="accuracy (validation)",
    ),
]


def model_fn(dictionary_dimension, hparams, seed):
    """Returns my model with the given hyper-parameters.
    Args:
      dictionary_dimension: Length of my dictionary to set up my neuronal network
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
        construct dropout layers in the model).
    Returns:
      A compiled Keras model.
    """
    random.Random(seed)
    model = tf.keras.models.Sequential()

    # I commented out my former models to keep trace

    # model.add(tf.keras.layers.SimpleRNN(dictionary_dimension))
    # model.add(tf.keras.layers.Dense(dictionary_dimension, activation='softmax'))


    # model.add(tf.keras.layers.Embedding(
    #     dictionary_dimension,
    #     hparams[HP_SEQUENCE_SIZE]-1,
    #     input_length=hparams[HP_SEQUENCE_SIZE]-1)
    # )
    # model.add(tf.keras.layers.LSTM(hparams[HP_HIDDEN_LAYERS], return_sequences=True))
    # model.add(tf.keras.layers.LSTM(hparams[HP_HIDDEN_LAYERS]))
    # model.add(tf.keras.layers.Dense(hparams[HP_HIDDEN_LAYERS], activation='relu'))
    # model.add(tf.keras.layers.Dense(dictionary_dimension, activation='softmax'))

    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            240,
            activation="relu"
        ),
        input_shape=(40-1, dictionary_dimension)
    ))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(dictionary_dimension))
    model.add(tf.keras.layers.Activation('softmax'))


    # OPTIMIZERS

    if hparams[HP_OPTIMIZER] == "adam":
        opt = tf.keras.optimizers.Adam(lr=hparams[HP_LEARNING_RATE], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif hparams[HP_OPTIMIZER] == "adagrad":
        opt = tf.keras.optimizers.Adagrad(lr=0.1)
    elif hparams[HP_OPTIMIZER] == "sdg":
        opt = tf.keras.optimizers.SGD(lr=hparams[HP_LEARNING_RATE], decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["categorical_accuracy"],
    )
    return model


def run(base_logdir, session_id, hparams, dictionary_dimension, preprocessor):
    """Run a training/validation session.
    Flags must have been parsed for this function to behave.
    Args:
      data: The data as loaded by `prepare_data()`.
      base_logdir: The top-level logdir to which to write summary data.
      session_id: A unique string ID for this session.
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      dictionary_dimension: Length of my dictionary to set up my neuronal network
      preprocessor: the preprocessor used to generate the data.
    """
    model = model_fn(dictionary_dimension=dictionary_dimension, hparams=hparams, seed=session_id)
    logdir = os.path.join(base_logdir, session_id)

    callback = tf.keras.callbacks.TensorBoard(
        logdir,
        update_freq=flags.FLAGS.summary_freq,
        profile_batch=0,  # workaround for issue #2084
    )
    hparams_callback = hp.KerasCallback(logdir, hparams)
    my_training_batch_generator = BatchGenerator(hparams[HP_BATCH_SIZE], preprocessor, is_validation=False)
    my_validation_batch_generator = BatchGenerator(hparams[HP_BATCH_SIZE], preprocessor, is_validation=True)
    result = model.fit_generator(generator=my_training_batch_generator,
                                 steps_per_epoch=np.ceil(len(my_training_batch_generator)/hparams[HP_BATCH_SIZE]),
                                 epochs=flags.FLAGS.num_epochs,
                                 verbose=1,
                                 validation_data=my_validation_batch_generator,
                                 validation_steps=np.ceil(len(my_validation_batch_generator)/hparams[HP_BATCH_SIZE]),
                                 callbacks=[callback,
                                            hparams_callback],
                                 )

    model.save(os.path.join(logdir, 'model.h5')) # Saving the model in its dir for I dont want to train it twice !


def run_all(logdir, verbose=False):
    """Perform random search over the hyperparameter space.
    Arguments:
      logdir: The top-level directory into which to write data. This
        directory should be empty or nonexistent.
      verbose: If true, print out each run's name as it begins.
    """
    rng = random.Random(0)
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)
    hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
    sessions_per_group = 1
    num_sessions = flags.FLAGS.num_session_groups * sessions_per_group
    session_index = 0  # across all session groups
    for group_index in xrange(flags.FLAGS.num_session_groups):
        hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
        hparams_string = str(hparams)
        preprocessor = Preprocessor(4,'') # We create an empty shell object
        preprocessor.load('preprocessor_save/full_tfidf.pre') # We load the real preprocessor used for the data
        preprocessor.sequence_size = 5 # We set sequence size
        for repeat_index in xrange(sessions_per_group):
            session_id = str(session_index)
            session_index += 1
            if verbose:
                print(
                    "--- Running training session %d/%d"
                    % (session_index, num_sessions)
                )
                print(hparams_string)
                print("--- repeat #: %d" % (repeat_index + 1))
            run(
                base_logdir=logdir,
                session_id=session_id,
                hparams=hparams,
                dictionary_dimension=len(preprocessor.dictionary),
                preprocessor=preprocessor
            )


def main(unused):
    np.random.seed(42)
    logdir = flags.FLAGS.logdir
    shutil.rmtree(logdir, ignore_errors=True)
    print("Saving output to %s." % logdir)
    run_all(logdir=logdir, verbose=True)
    print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
    app.run(main)
