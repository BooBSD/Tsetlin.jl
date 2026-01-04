# Source: https://github.com/cair/tmu/blob/main/examples/classification/IMDbTextCategorizationDemo.py
# License: MIT © 2023 Centre for Artificial Intelligence Research (CAIR)

import argparse
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer


def main(args):

    train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
    train_x, train_y = train
    test_x, test_y = test

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    print("Producing bit representation...", end=" ")

    id_to_word = {value: key for key, value in word_to_id.items()}

    training_documents = []
    for i in range(train_y.shape[0]):
        terms = []
        for word_id in train_x[i]:
            terms.append(id_to_word[word_id].lower())
#            terms.append(id_to_word[word_id])

        training_documents.append(terms)

    testing_documents = []
    for i in range(test_y.shape[0]):
        terms = []
        for word_id in test_x[i]:
            terms.append(id_to_word[word_id].lower())
#            terms.append(id_to_word[word_id])

        testing_documents.append(terms)

    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
#        lowercase=True,
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)
    print("Done.")

    print("Selecting Features....", end=" ")

    SKB = SelectKBest(chi2, k=args.features)
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train).toarray().astype(np.uint32)
    X_test = SKB.transform(X_test).toarray().astype(np.uint32)

    output_test = np.c_[X_test, Y_test]
    np.savetxt("/tmp/IMDBTestData.txt", output_test, fmt="%d")

    output_train = np.c_[X_train, Y_train]
    np.savetxt("/tmp/IMDBTrainingData.txt", output_train, fmt="%d")

    print("Done.")


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ngram", default=4, type=int)
    parser.add_argument("--features", default=12800, type=int)
    parser.add_argument("--imdb-num-words", default=40000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    results = main(default_args())
