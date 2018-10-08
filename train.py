"""Create the Markov models for each language in the training corpus.

Author: Joe Comer
"""


import pandas as pd
import os
import unicodedata


def count_bigram(bigram, string):
    """Count bigram occurences in a string.

    This will actually also count unigrams.
    """
    count = start = 0
    while True:
        start = string.find(bigram, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def remove_accents(input_str):
    """Remove any diacritical marks from text."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return str(only_ascii)[2:-3].lower()


def filter_string(string):
    """Encode the training/test data.

    Translate the raw language document into its
    corresponding phoneme encoding.
    """
    plosives = ['p', 't', 'k', 'b', 'd', 'g', 'c', 'q']
    fricatives = ['f', 'g', 'h', 'j', 'l', 'r', 's', 'v', 'w', 'y', 'z']
    nasals = ['n', 'm']
    vowels = ['a', 'e', 'i', 'o', 'u']

    filter_dict = dict()
    for letter in plosives:
        filter_dict[letter] = '1'
    for letter in fricatives:
        filter_dict[letter] = '2'
    for letter in nasals:
        filter_dict[letter] = '3'
    for letter in vowels:
        filter_dict[letter] = '4'

    # Special case for x, ng, ch, ph, sh, th
    filter_dict['x'] = '12'
    filter_dict['ng'] = '3'
    filter_dict['ch'] = '2'
    filter_dict['ph'] = '2'
    filter_dict['sh'] = '2'
    filter_dict['th'] = '2'

    # Convert all numbers to 0.
    for i in range(10):
        filter_dict[str(i)] = '0'

    out_string = ""
    i = 0
    while i < len(string):
        if string[i] not in filter_dict:
            if string[i] == " ":
                filtered = " "
            else:
                # Non-alphanumeric characters will receive the '?'
                # wildcard, along with any characters not seen in the
                # training corpus.
                filtered = "?"
        elif (i < len(string) - 1) and (string[i:i + 2] in filter_dict):
            sound = string[i:i + 2]
            filtered = filter_dict[sound]
            i += 1
        else:
            sound = string[i]
            filtered = filter_dict[sound]
        out_string += filtered
        i += 1
    return out_string


def create_models():
    """Create the probabilities tables.

    Create probabilities tables for the Markov models for each language
    in the training folder. Called by predict.py
    """

    def train_model(filename):
        """Train a bigram model on a text file.

        filename: name of the text file
        returns: probs, dataframe of conditional probabilities
        The i,jth entry of probs is P(j|i)
        """
        def prob(i, j, string):
            """Calculate the probability of j given i."""
            first_char = str(i)
            second_char = str(j)
            conditional_prob = (bigram_counts[first_char + second_char] +
                                (unigram_counts[second_char] / len(string))) /\
                                 (unigram_counts[first_char] + 1)
            return conditional_prob
        with open("./Language_Identification/train/Filtered/" + filename) as iFile:
            # These files are all a single line after the filtering above.
            corpus = next(iFile)

            # Add a start character and end character to ensure a single
            # probability distribution over documents of all lengths.
            corpus = "." + corpus + "!"
            charset = [str(x) for x in set(corpus)]
            bigrams = [x + y for x in charset for y in charset]
            bigram_counts = dict([(big, count_bigram(big, corpus))
                                  for big in bigrams])
            unigram_counts = dict([(uni, count_bigram(uni, corpus))
                                   for uni in charset])
            probs = pd.DataFrame(columns=[[char for char in charset]])
            for char in charset:
                probs.loc[char] = [prob(char, j, corpus)
                                   for j in probs.columns]
        return probs

    # Create phoneme-encoded corpus files for training.
    for root, dirs, files in os.walk("./Language_Identification/train/"):
        for filename in files:
            if filename[:4] != "filt":
                with open("./Language_Identification/train/" + filename, 'r+',
                          encoding='UTF-8') as iFile:
                    with open("./Language_Identification/train/Filtered/filtered_" +
                              filename, 'w+') as oFile:
                        for line in iFile:
                            oFile.write(filter_string(remove_accents(line)))

    models = dict()
    for root, dirs, files in os.walk("./Language_Identification/train/"):
        for filename in files:
            if filename[:4] == "filt":
                models[filename[9:]] = train_model(filename)

    return models
