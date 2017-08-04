import collections
import itertools
import os
import os.path
import pickle
import re
import string
import sys

import numpy as np
from nltk import conlltags2tree, pos_tag, tree2conlltags, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ner_tags = collections.Counter()

basepath = os.path.dirname(__file__)
corpus_root = os.path.abspath(os.path.join(basepath, "gmb-2.2.0"))

# corpus_root = "gmb-2.2.0.zip"
# reload(sys)
# sys.setdefaultencoding('utf-8')


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    # file_handle = zipfile.ZipFile('gmb-2.2.0.zip', 'r')
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [
                            seq for seq in annotated_sentence.split('\n') if seq]
                        standard_form_tokens = []
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[
                                0], annotations[1], annotations[3]
                            ner_tags[ner] += 1
                            # Get only the primary category
                            if ner != 'O':
                                ner = ner.split('-')[0]

                            # if tag in ('LQU', 'RQU'):
                            #     tag = "``"

                            standard_form_tokens.append((word, tag, ner))
                        conll_tokens = to_conll_iob(standard_form_tokens)

                        yield conlltags2tree(conll_tokens)
    print("Data read done")
    # yield [((w, t), iob) for w, t, iob in conll_tokens]


def to_conll_iob(annotated_sentence):

    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))

    return proper_iob_tokens

stemmer = SnowballStemmer('english')


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # init the stemmer

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + \
        list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[-1]
    prevpreviob = history[-2]

    feat_dict = {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'shape': shape(word),

        'next-word': nextword,
        'next-pos': nextpos,
        'next-lemma': stemmer.stem(nextword),
        'next-shape': shape(nextword),

        'next-next-word': nextnextword,
        'next-next-pos': nextnextpos,
        'next-next-lemma': stemmer.stem(nextnextword),
        'next-next-shape': shape(nextnextword),

        'prev-word': prevword,
        'prev-pos': prevpos,
        'prev-lemma': stemmer.stem(prevword),
        'prev-iob': previob,
        'prev-shape': shape(prevword),

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
        'prev-prev-lemma': stemmer.stem(prevprevword),
        'prev-prev-iob': prevpreviob,
        'prev-prev-shape': shape(prevprevword),
    }

    return feat_dict


def shape(word):
    word_shape = 'other'
    if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        word_shape = 'number'
    elif re.match('\W+$', word):
        word_shape = 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        word_shape = 'capitalized'
    elif re.match('[A-Z]+$', word):
        word_shape = 'uppercase'
    elif re.match('[a-z]+$', word):
        word_shape = 'lowercase'
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'

    return word_shape


def to_dataset(parsed_sentences, feature_detector):
    X, y = [], []
    for parsed in parsed_sentences:
        iob_tagged = tree2conlltags(parsed)
        words, tags, iob_tags = list(zip(*iob_tagged))

        tagged = list(zip(words, tags))

        for index in range(len(iob_tagged)):
            X.append(feature_detector(
                tagged, index, history=iob_tags[:index]))
            y.append(iob_tags[index])

    return X, y

# for new sentences
# def parse(self, tokens):
#         """
#         Chunk a tagged sentence
#         :param tokens: List of words [(w1, t1), (w2, t2), ...]
#         :return: chunked sentence: nltk.Tree
#         """
#         history = []
#         iob_tagged_tokens = []
#         for index, (word, tag) in enumerate(tokens):
#             iob_tag = self._classifier.predict([self._feature_detector(tokens, index, history)])[0]
#             history.append(iob_tag)
#             iob_tagged_tokens.append((word, tag, iob_tag))
#
#         return conlltags2tree(iob_tagged_tokens)
#

if __name__ == "__main__":
    data = read_gmb(corpus_root)
    train_data = itertools.islice(data, 100)
    X, y = to_dataset(train_data, features)
    # print(X)
    # print(y)
    # Split the data into 70% training data and 30% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # print(X_train)
    # print(y_train)
    # Transform dictionary into list

    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(X_train)
    print(X)

    # print(vectorizer.get_feature_names())

    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

    # Train the perceptron
    print("Training\n")
    ppn.fit(X, y_train)
    print("Train complete")

    # Test data
    x_test = vectorizer.transform(X_test)

    y_pred = ppn.predict(x_test)

    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(ppn, 'ppn.pkl')

    # vec_ppn = {'vectorizer', vectorizer, 'ppn', ppn}
    # joblib.dump(vec_ppn, 'vec_and_ppn.pkl')

    # filename = 'aug_joblib.sav'
    # joblib.dump(ppn, filename)
    #
    # # pickle.dump(ppn, open(filename, 'wb'))
    # # pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))

    print("Predicting new sentences:")

    # create tuple of given sentneces
    # new = pos_tag(word_tokenize("I live in Bhaktapur."))
    new = pos_tag(word_tokenize(
        "A person died and four others were injured when a micro bus hit them at Jorpati, Kathmandu on Thursday."))

    # print(new)
    history = []
    iob_tagged_tokens = []
    for index, (word, tag) in enumerate(new):
        f = features(new, index, history)
        new_transform = vectorizer.transform(f)
        ner_tag = ppn.predict(new_transform)
        # print("Predicted NER TAG for current")
        # print(word, "Corresponding Tokens", ner_tag[0])
        # print(ppn.predict(new_transform))

        iob_tag = ner_tag[0]
        history.append(iob_tag)
        # print(iob_tag)
        # print("\n")
        # print("Predict current one\n")
        #
        #
        # new_transform = vectorizer.transform(iob_tag)
        # print(ppn.predict(new_transform))

        iob_tagged_tokens.append((word, tag, iob_tag))

    # print(ready)

    print(iob_tagged_tokens)
