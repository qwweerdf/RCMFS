import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import ast
import pandas as pd
import numpy as np
from sklearn import svm
import re
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
nltk.download('averaged_perceptron_tagger')
from flair.data import Sentence
from flair.nn import Classifier


def feature_extraction(data, ner=False):
    if ner:
        tagger = Classifier.load('ner-fast')
    data_list = data.content.values.tolist()
    tokens = []
    for each in data_list:
        tokens.append(word_tokenize(each))

    # feature 1: the count of NNP
    # feature 2: count of JJ + NN + NNS + IN + VB + VBD:
    # feature 3: numbers only
    # feature 4: doi
    # feature 5: pages
    # feature 6: year
    # feature 7: if volume
    # feature 8: if issue count
    feat1 = []
    feat2 = []
    feat3 = []
    feat4 = []
    feat5 = []
    feat6 = []
    feat7 = []
    feat8 = []
    t1 = []
    t2 =[]
    for i, each in enumerate(tokens):
        tags = pos_tag(each)
        tags = np.array(tags)

        if ner:
            # make a sentence
            sentence = Sentence(data_list[i])

            # run NER over sentence
            tagger.predict(sentence)

            # print the sentence with all annotations
            count = 0
            for label in sentence.labels:
                if label.value == 'PER':
                    count += 1
            feat1.append(count)
        else:
            NNP_count = np.sum(tags[:, 1] == 'NNP')
            feat1.append(NNP_count)
        DT_count = np.sum(tags[:, 1] == 'DT')
        TO_count = np.sum(tags[:, 1] == 'TO')
        EX_count = np.sum(tags[:, 1] == 'EX')
        CC_count = np.sum(tags[:, 1] == 'CC')
        JJ_count = np.sum(tags[:, 1] == 'JJ')
        RB_count = np.sum(tags[:, 1] == 'RB')
        WDT_count = np.sum(tags[:, 1] == 'WDT')
        RBR_count = np.sum(tags[:, 1] == 'RBR')
        JJR_count = np.sum(tags[:, 1] == 'JJR')
        VB_count = np.sum(tags[:, 1] == 'VB')
        VBZ_count = np.sum(tags[:, 1] == 'VBZ')
        VBG_count = np.sum(tags[:, 1] == 'VBG')
        VBD_count = np.sum(tags[:, 1] == 'VBD')
        VBN_count = np.sum(tags[:, 1] == 'VBN')
        VBP_count = np.sum(tags[:, 1] == 'VBP')
        sent_count = DT_count + TO_count + EX_count + CC_count + JJ_count + RB_count + WDT_count + RBR_count + JJR_count + VB_count + VBD_count + VBG_count + VBZ_count + VBN_count + VBP_count
        feat2.append(sent_count)

        if len(each) < 5:
            if any(element.isdigit() for element in each):
                feat3.append(1)
            else:
                feat3.append(0)
        else:
            feat3.append(0)

        if re.match(r'(http:\/\/dx.doi.org\/)?10\.\d{4,9}(\/.+)+', each[0]) is not None:
            feat4.append(1)
        else:
            feat4.append(0)

        if re.match(r'^\d+-+\d+$', each[0]) is not None:
            feat5.append(1)
        elif re.match(r'^e\d+$', each[0]) is not None:
            feat5.append(1)
        elif len(each) == 1 and each[0].isdigit() and int(each[0]) > 100:
            feat5.append(1)
        else:
            feat5.append(0)

        def contains_year(tkn):
            for each in tkn:
                pattern = r".*\b(18\d{2}|19\d{2}|20[01]\d|202[0-5])\b.*"
                match = re.search(pattern, each)
                if match is not None:
                    return 3
                continue
            return 0

        exist_year = contains_year(each)
        feat6.append(exist_year)

        if re.match(r'\d+', each[0]) is not None:
            value = int(re.search(r'\d+', each[0]).group())
            if value > 20:
                feat7.append(1)
                feat8.append(0)
            else:
                feat7.append(0)
                feat8.append(1)
        else:
            feat7.append(0)
            feat8.append(0)

    return feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8

def transform(data, ner=False):
    feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8 = feature_extraction(data, ner)
    ref_data = data.drop('content', axis=1)
    ref_data.reset_index(drop=True, inplace=True)

    ref_data['nnp'] = feat1
    ref_data['sent'] = feat2
    ref_data['num'] = feat3
    ref_data['doi'] = feat4
    ref_data['pages'] = feat5
    ref_data['year'] = feat6
    ref_data['volume'] = feat7
    ref_data['issue'] = feat8

    x = ref_data.iloc[:, 1:].values
    y = ref_data['type'].values

    return x, y


def train(x, y):
    accuracies = []
    balanced_accuracy_scores = []
    roc_scores = []
    pr_scores = []
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

        # Create a logistic regression model
        model = svm.SVC(probability=True)

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Calculate the accuracy of the model for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        balanced_accuracy_score_ = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracy_scores.append(balanced_accuracy_score_)

        pred_proba = model.predict_proba(X_test)
        roc_score = roc_auc_score(y_test, pred_proba, multi_class='ovr')
        roc_scores.append(roc_score)

        average_precisions = []
        for class_idx in range(8):
            y_true_class = (y_test == class_idx)  # True for samples of the current class, False otherwise
            y_scores_class = pred_proba[:, class_idx]  # Predicted probabilities for the current class
            average_precisions.append(average_precision_score(y_true_class, y_scores_class))
        pr_scores.append(sum(average_precisions) / len(average_precisions))

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

    f1_scores = 2 * (np.array(roc_scores) * np.array(pr_scores)) / (np.array(roc_scores) + np.array(pr_scores))
    average_f1_score = sum(f1_scores) / num_folds
    print("Average f1 score:", average_f1_score)
    best_performance_index = f1_scores.tolist().index(max(f1_scores))

    return models[best_performance_index]


def execute():
    def file_reader():
        ref_tags = []
        with open('train.txt') as file:
            for line in file:
                ref_tags.append(ast.literal_eval(line))

        return ref_tags

    def list2pd(ref):
        ref = [element for sublist in ref for element in sublist]
        return pd.DataFrame(ref, columns=['content', 'type'])

    train_ref = file_reader()
    data = list2pd(train_ref)
    data['type'] = data['type'].replace(
        {'authors': 0, 'title': 1, 'volume': 2, 'issue': 3, 'pages': 4, 'journal': 5, 'year': 6, 'doi': 7})
    data.loc[data['type'] == 1, 'content'] = data.loc[data['type'] == 1, 'content'].str.lower()
    # data cleaning
    data = data.replace('', np.nan)

    data = data.dropna()
    data.reset_index(drop=True, inplace=True)

    # transform and train
    x, y = transform(data, ner=True)
    best_model = train(x, y)

    # store data
    with open('/Users/jialong/PycharmProjects/RCMFS/svm_component_identification.pkl', 'wb') as file:
        pickle.dump(best_model, file)


if __name__ == '__main__':
    execute()

