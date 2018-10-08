# Language Identification via Phoneme encoding

We propose to distinguish languages by maximum likelihood via a Markov modeling of each language of interest, after encoding the raw text into phonemme encodings. Phonemes are broken down into consonant types and vowels.

## To run:
Simply run predict.py. New files can be added either to the train or test folders, but
the model is currently limited to languages which employ variations on the Latin alphabet.

## Requirements:
pandas
numpy
unicodedata
