import random
import subprocess
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import scipy
import nlpaug.augmenter.word as naw
from imblearn.over_sampling import SMOTE
import sklearn_crfsuite
import re
import bibtexparser
import ast
import nltk
import numpy
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from nltk import pos_tag

"""
CRF is tested for component identification, but the performance is not ideal. 
Majorly because of the bias of the dataset i.e. too much authors and titles.
"""


if __name__ == '__main__':

    nltk.download('punkt')  # Download the Punkt tokenizer
    sentences = []
    tags = []
    with open("../../component_identification/train.txt", "r") as file:
        for line in file:
            instance = ast.literal_eval(line.strip())
            temp = []
            tokens = []
            for item in instance:
                if item[0] == '':
                    continue
                each_tkn = nltk.word_tokenize(item[0])
                tokens.extend(nltk.word_tokenize(item[0]))

                for each in each_tkn:
                    temp.append(item[1])
            sentences.append(tokens)
            tags.append(temp)


    def contains_year(tkn):
        pattern = r".*\b(18\d{2}|19\d{2}|20[01]\d|202[0-5])\b.*"
        match = re.search(pattern, tkn)
        if match is not None:
            return True
        return False


    # Specify the class weights you desire
    class_weights = {
        'authors': 0,
        'title': 0.5,
        'journal': 100.0,
        'doi': 100.0,
        'year': 100.0,
        'pages': 100.0,
        'volume': 100.0,
        'issue': 100.0,
    }
    # Let's define the features for each token in your training set
    def features(sentence, index):
        # sentence[index] is the token in the sentence. index is the index of the token in the sentence.
        token = sentence[index]

        return {
            'token': token,
            'index': index,
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': token[0].upper() == token[0],
            'is_all_caps': token.upper() == token,
            'is_all_lower': token.lower() == token,
            'prefix-1': token[0],
            'suffix-1': token[-1],
            'prev_token': '' if index == 0 else sentence[index - 1],
            'next_token': '' if index == len(sentence) - 1 else sentence[index + 1],
            'is_digit': token.isdigit(),
            'is_year': contains_year(token),
            'is_doi': re.match(r'(http:)?(\/\/dx.doi.org\/)?10\.\d{4,9}(\/.+)+', token) is not None,
            'is_http': re.match(r'http', token) is not None,
            'is_colon': re.match(r':', token) is not None,
            'is_nnp': pos_tag([token])[0][1] == 'NNP',
            'is_nn': pos_tag([token])[0][1] == 'NN',
            'is_page': re.match(r'^\d+(-+\d+)?$', token) is not None,
            'weight': class_weights.get(token, 1)
        }

    # The feature representation of the sentence
    def transform_to_dataset(sentences, tags):
        X, y = [], []
        for sentence_idx in range(len(sentences)):
            X.append([features(sentences[sentence_idx], index) for index in range(len(sentences[sentence_idx]))])
            y.append(tags[sentence_idx])
        # print(X)
        # print()
        return X, y





    """
    this is oversampling DONT WORK!!!
    """
    # Flatten the labels to count occurrences
    flat_labels = [label for sublist in tags for label in sublist]
    label_counts = Counter(flat_labels)
    print(label_counts)
    # # Determine minority labels
    # # For simplicity, let's assume any label occurring less than max count is a minority
    # max_count = label_counts.most_common(1)[0][1]
    # minority_labels = {item[0] for item in label_counts.items() if item[1] < max_count}
    #
    # # Find sequences containing minority labels
    # minority_sequences_indices = {i for i, label_sequence in enumerate(tags) for label in label_sequence if
    #                               label in minority_labels}
    #
    # # Oversample: duplicate each minority sequence once
    # # Note: Depending on how imbalanced your data is, you might want to duplicate more or fewer times
    # for i in minority_sequences_indices:
    #     sentences.append(sentences[i])
    #     tags.append(tags[i])
    #
    # # Shuffle the data if needed
    # combined = list(zip(sentences, tags))
    # random.shuffle(combined)
    # sentences[:], tags[:] = zip(*combined)
    # print(Counter([label for sublist in tags for label in sublist]))




    """
    oversampling 2
    """



    # Choose the transformation technique
    # augmenter = naw.ContextualWordEmbsAug()
    #
    # # Augment the data
    # X_augmented = []
    # y_augmented = []
    #
    # for x_seq, y_seq in zip(sentences, tags):
    #     for _ in range(1):  # You can adjust the number of augmentations per sequence
    #         augmented_text = augmenter.augment(x_seq)
    #         X_augmented.append(augmented_text)
    #         y_augmented.append(y_seq)
    #
    # # Combine original and augmented data
    # X_combined = sentences + X_augmented
    # y_combined = tags + y_augmented
    #
    # # Shuffle the combined data
    # combined = list(zip(X_combined, y_combined))
    # random.shuffle(combined)
    # X_combined, y_combined = zip(*combined)
    #
    # print(y_combined)
    #
    # print(Counter([label for sublist in y_combined for label in sublist]))






    X_train, y_train = transform_to_dataset(sentences, tags)
    print(X_train)
    print(y_train)
    # Initialize CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=1000,
        all_possible_transitions=True
    )

    # params_space = {
    #     'c1': scipy.stats.expon(scale=0.5),
    #     'c2': scipy.stats.expon(scale=0.05),
    # }

    # use the same metric for evaluation
    # f1_scorer = make_scorer(metrics.flat_f1_score,
    #                         average='weighted', labels=tags)
    #
    # # search
    # rs = RandomizedSearchCV(crf, params_space,
    #                         cv=3,
    #                         verbose=1,
    #                         n_jobs=-1,
    #                         n_iter=50,
    #                         scoring=f1_scorer)
    # rs.fit(X_train, y_train)
    # print('best params:', rs.best_estimator_)
    # print('best CV score:', rs.best_score_)
    # print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    # Train the CRF model
    crf.fit(X_train, y_train)
    single = nltk.word_tokenize(
        "F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. http://dx.doi.org/10.1093/ajae/aaq063")

    new_sentence = single
    X_test = [features(new_sentence, index) for index in range(len(new_sentence))]
    print(X_test)
    y_pred = crf.predict_single(X_test)
    print(single)
    print(y_pred)
    for i in range(len(single)):
        print(single[i])
        print(y_pred[i])
        print()
    # res = []
    # res.append(single)
    # res.append(y_pred)
    # print(crf.predict(X_train))
    # print(crf.score(X_train, y_train))
