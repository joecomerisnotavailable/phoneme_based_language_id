"""Predict language of text files in test folder.

Author: Joe Comer
"""

import numpy as np
import os
import operator
from train import create_models, filter_string, remove_accents


models = create_models()


def language_id(filename, models_dict=models):
    """Identify a language from a text file."""
    with open("./Language_Identification/test/" + filename,
              "r+", encoding='UTF-8') as iFile:
        corpus = ""
        for line in iFile:
            corpus += filter_string(remove_accents(line))
        corpus = "." + corpus + "!"
        likelihoods = dict([(model, 0) for model in models_dict])
        for i in range(len(corpus) - 1):
            bigram = corpus[i:i + 2]
            for model in models_dict:
                likelihoods[model] += np.log(
                    models_dict[model][bigram[1]][bigram[0]])
        most_likely = max(likelihoods.items(), key=operator.itemgetter(1))[0]
    return most_likely


for root, dirs, files in os.walk("./Language_Identification/test/"):
    for filename in files:
        print("File:\t", "Prediction:\n",
              filename + "\t", language_id(filename))
