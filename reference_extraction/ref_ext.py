import pickle
import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from flair.data import Sentence
from flair.nn import Classifier

# feature extraction
def feature_extraction(data):
    # tagger = Classifier.load('ner-fast')

    data_list = data.content.values.tolist()
    tokens = []
    for each in data_list:
        tokens.append(word_tokenize(each))

    # feature 1: the amount of NNP
    # feature 2: the amount of VB and VBD
    # feature 3: comma count
    # feature 4: dot count
    # feature 5: digit count
    # feature 6: year or not
    # feature 7: amount of DT+TO+EX+CC+JJ+RB+WDT+RBR+JJR
    # feature 8: amount of MD
    # feature 9: text length
    feat1 = []
    feat2 = []
    feat3 = []
    feat4 = []
    feat5 = []
    feat6 = []
    feat7 = []
    feat8 = []
    feat9 = []
    for i, token in enumerate(tokens):
        tags = pos_tag(token)


        tags = np.array(tags)
        NNP_count = np.sum(tags[:, 1] == 'NNP')
        feat1.append(NNP_count)
        # NNP_count = np.sum(tags[:, 1] == 'NNP')
        # make a sentence
        # sentence = Sentence(data_list[i])
        #
        # # run NER over sentence
        # tagger.predict(sentence)
        #
        # # print the sentence with all annotations
        # print(sentence.labels)
        # count = 0
        # for label in sentence.labels:
        #     if label.value == 'PER':
        #         count += 1
        # feat1.append(count)

        VB_count = np.sum(tags[:, 1] == 'VB')
        VBZ_count = np.sum(tags[:, 1] == 'VBZ')
        VBG_count = np.sum(tags[:, 1] == 'VBG')
        VBD_count = np.sum(tags[:, 1] == 'VBD')
        VBN_count = np.sum(tags[:, 1] == 'VBN')
        VBP_count = np.sum(tags[:, 1] == 'VBP')
        feat2_count = VB_count + VBD_count + VBG_count + VBZ_count + VBN_count + VBP_count
        feat2.append(feat2_count)

        comma_count = token.count(',')
        feat3.append(comma_count)

        dot_count = token.count('.')
        feat4.append(dot_count)

        digit_count = sum(1 for char in token if char.isdigit())
        feat5.append(digit_count)

        def contains_year(tkn):
            for each in tkn:
                pattern = r"\b(18\d{2}|19\d{2}|20[01]\d|202[0-5])\b"
                match = re.search(pattern, each)
                if match is not None:
                    return 1
                continue
            return 0

        exist_year = contains_year(token)
        feat6.append(exist_year)

        DT_count = np.sum(tags[:, 1] == 'DT')
        TO_count = np.sum(tags[:, 1] == 'TO')
        EX_count = np.sum(tags[:, 1] == 'EX')
        CC_count = np.sum(tags[:, 1] == 'CC')
        JJ_count = np.sum(tags[:, 1] == 'JJ')
        RB_count = np.sum(tags[:, 1] == 'RB')
        WDT_count = np.sum(tags[:, 1] == 'WDT')
        RBR_count = np.sum(tags[:, 1] == 'RBR')
        JJR_count = np.sum(tags[:, 1] == 'JJR')
        feat7_count = DT_count + TO_count + EX_count + CC_count + JJ_count + RB_count + WDT_count + RBR_count + JJR_count
        feat7.append(feat7_count)

        feat8.append(np.sum(tags[:, 1] == 'MD'))

        lengths = [len(s) for s in token]
        feat9.append(sum(lengths))
    return feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9


def transform(data):
    feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9 = feature_extraction(data)
    data['nnp'] = feat1
    data['vb'] = feat2
    data['comma'] = feat3
    data['dot'] = feat4
    data['digit'] = feat5
    data['year'] = feat6
    data['sent'] = feat7
    data['md'] = feat8
    data['length'] = feat9
    data.drop('content', axis=1)

    for feat in ['nnp', 'vb', 'comma', 'dot', 'digit', 'year', 'sent', 'md', 'length']:
        print('{} not reference mean: {}'.format(feat, np.mean(data[feat].loc[data['label'] == 0])))
        print('{} not reference var: {}'.format(feat, np.var(data[feat].loc[data['label'] == 0])))
        print('{} is reference mean: {}'.format(feat, np.mean(data[feat].loc[data['label'] == 1])))
        print('{} is reference var: {}'.format(feat, np.var(data[feat].loc[data['label'] == 1])))
        print()

    x = data.iloc[:, 2:].values
    y = data['label'].values

    return x, y


def train(x, y):
    accuracies = []
    balanced_accuracy_scores = []
    roc_scores = []
    pr_scores = []
    f1_scores = []
    models = []

    # Define the number of folds
    num_folds = 10

    # Initialize the stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(x, y):
        # Split the data into training and test sets for this fold
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create an SVM model
        model = svm.SVC()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        balanced_accuracy_score_ = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracy_scores.append(balanced_accuracy_score_)

        roc_score = roc_auc_score(y_test, y_pred)
        roc_scores.append(roc_score)

        pr_score = average_precision_score(y_test, y_pred)
        pr_scores.append(pr_score)

        f1_score_ = f1_score(y_test, y_pred)
        f1_scores.append(f1_score_)

        models.append(model)

    # Calculate the average accuracy across all folds
    average_accuracy = sum(accuracies) / num_folds
    print("Average Accuracy:", average_accuracy)

    average_balanced_accuracy_scores = sum(balanced_accuracy_scores) / num_folds
    print("Average balanced Accuracy:", average_balanced_accuracy_scores)

    average_roc_score = sum(roc_scores) / num_folds
    print("Average roc score:", average_roc_score)

    average_pr_score = sum(pr_scores) / num_folds
    print("Average pr score:", average_pr_score)

    average_f1_score = sum(f1_scores) / num_folds
    print("Average f1 score:", average_f1_score)
    best_performance_index = pr_scores.index(max(pr_scores))

    return models[best_performance_index]


def execute():
    # read data
    raw = []
    with open('reference_extraction/corpus.txt') as file:
        for row in file:
            row = row.replace('\n', '')
            row = row.replace('</fnote>', '')
            row = row.replace('<fnote>', '')
            raw.append(row)

    # data preparation
    data = pd.DataFrame(raw, columns=['content'])
    data['label'] = 1
    data['label'].iloc[250:] = 0

    # feature extraction and train
    x, y = transform(data)
    best_model = train(x, y)

    # store data
    with open('reference_extraction/svm_reference_extraction.pkl', 'wb') as file:
        pickle.dump(best_model, file)


if __name__ == '__main__':
    execute()
