from Preprocessor import Preprocessor

"""
This script is used to train a predictor separately from model
"""
preprocessor = Preprocessor(5, data_path='data/out.txt', index_rather_than_vector=False)
preprocessor.train()
preprocessor.save('preprocessor_save/full.pre')
print(len(preprocessor.dictionary))  # Always intersting to know the dictionary size