import os

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
import ast
import pandas as pd
import numpy as np
from sklearn import svm
import re
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score, \
    confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
nltk.download('averaged_perceptron_tagger')
from flair.data import Sentence
from flair.nn import Classifier

tagger = Classifier.load('ner-fast')

"""
Train Feature Extraction Models for component identification
"""


def feature_extraction(data, ner=False):
    """
    extract features from text to numeric features
    :param data: tabular text data
    :param ner: if Flair NER
    :return: features
    """

    data_list = data.content.values.tolist()
    tokens = []
    for each in data_list:
        tokens.append(word_tokenize(each))

    # feature 1: the count of NNP
    # feature 2: count of JJ + NN + NNS + IN + VB + VBD + ...:
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
    t2 = []
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

        # if the token is not a year and is a digit and not doi and not pages and not exceed to 100
        if re.match(r'^\d+$', each[0]) is not None and contains_year(each) == 0 and re.match(r'(http:\/\/dx.doi.org\/)?10\.\d{4,9}(\/.+)+', each[0]) is None and re.match(r'^\d+-+\d+$', each[0]) is None and int(each[0]) <= 100:
            value = int(re.search(r'\d+', each[0]).group())
            # error handling for the first and last token
            if i != 0 and i != len(tokens) - 1:
                # if previous token is a digit number, it means that it is more possible an issue number and also avoid large page number
                if re.match(r'^\d+$', tokens[i-1][0]) is not None and re.match(r'^\d+-+\d+$', tokens[i-1][0]) is None and int(tokens[i-1][0]) <= 100:
                    feat7.append(0)
                    feat8.append(1)
                # if next token is a digit number, it means that it is more possible a volume number
                elif re.match(r'^\d+$', tokens[i+1][0]) is not None and re.match(r'^\d+-+\d+$', tokens[i+1][0]) is None and int(tokens[i+1][0]) <= 100:
                    feat7.append(1)
                    feat8.append(0)
                # if neither previous nor next token are digit number, then set threshold 20 between volume and issue and also avoid large page number
                else:
                    if value > 20:
                        feat7.append(1)
                        feat8.append(0)
                    else:
                        feat7.append(0)
                        feat8.append(1)
            else:
                feat7.append(0)
                feat8.append(0)
        else:
            feat7.append(0)
            feat8.append(0)

    return feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8


def transform(data, ner=False):
    """
    transform data to x and y
    :param data: tabular text data
    :param ner: if Flair NER
    :return: numeric features x and label y
    """

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


def train(x, y, model_type='svm'):
    """
    train models to fit the data
    :param x: numeric features x
    :param y: numeric label y
    :param model_type: svm
    :return: the best model
    """

    accuracies = []
    balanced_accuracy_scores = []
    roc_scores = []
    pr_scores = []
    models = []
    cms = []
    # Define the number of folds
    num_folds = 10

    # Initialize the stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # Perform k-fold cross-validation
    for train_index, test_index in skf.split(x, y):
        # Split the data into training and test sets for this fold
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create a model
        if model_type == 'svm':
            print('using svm')
            model = svm.SVC(probability=True)
        elif model_type == 'brf':
            print('using brf')
            model = RandomForestClassifier(n_estimators=100)  # 100 trees in the forest
        else:
            return "the model given is not recognised or not supported!"
        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)
        cms.append(confusion_matrix(y_test, y_pred))

        # Calculate the accuracy of the model for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        balanced_accuracy_score_ = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracy_scores.append(balanced_accuracy_score_)

        pred_proba = model.predict_proba(X_test)

        # micro roc
        # average_rocs = []
        # for class_idx in range(8):
        #     y_true_class = (y_test == class_idx)  # True for samples of the current class, False otherwise
        #     y_scores_class = pred_proba[:, class_idx]  # Predicted probabilities for the current class
        #     average_rocs.append(roc_auc_score(y_true_class, y_scores_class, average='micro'))
        # roc_scores.append(sum(average_rocs) / len(average_rocs))

        # macro roc
        roc_score = roc_auc_score(y_test, pred_proba, average='macro', multi_class='ovo')
        roc_scores.append(roc_score)



        # micro pr
        # average_precisions = []
        # for class_idx in range(8):
        #     y_true_class = (y_test == class_idx)  # True for samples of the current class, False otherwise
        #     y_scores_class = pred_proba[:, class_idx]  # Predicted probabilities for the current class
        #     average_precisions.append(average_precision_score(y_true_class, y_scores_class, average='micro'))
        # pr_scores.append(sum(average_precisions) / len(average_precisions))

        # macro pr
        pr_score = precision_score(y_test, y_pred, average='macro')
        pr_scores.append(pr_score)




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

    # calculate confusion matrix
    avg_cm = np.sum(cms, axis=0) / len(cms)

    # plot confusion matrix
    sns.heatmap(np.round(avg_cm), annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # get the best model
    best_performance_index = f1_scores.tolist().index(max(f1_scores))

    return models[best_performance_index]


# architecture for the Neural Network
class TabularNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def train_nn(X, y):
    """
    train neural network
    :param X: x from transform function
    :param y: y from transform function
    :return: None
    """

    num_classes = 8
    batch_size = 32
    lr = 0.001
    epochs = 500

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    # Create datasets and dataloaders
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = TabularNN(input_dim=X.shape[1], hidden_dim=50, output_dim=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    eval_losses = []

    # main loop for training and validation
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for batch in train_loader:
            data, targets = batch
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets.long())
            train_loss += loss
            loss.backward()
            optimizer.step()
        current_train_loss = train_loss / len(train_loader)
        train_losses.append(current_train_loss.detach().numpy())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_train_loss}")

        model.eval()
        correct = 0
        total = 0
        eval_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                data, targets = batch
                outputs = model(data)
                loss = loss_fn(outputs, targets.long())
                eval_loss += loss
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            current_eval_loss = eval_loss / len(test_loader)
            eval_losses.append(current_eval_loss.detach().numpy())
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_eval_loss}")
            print(f"Eval accuracy: {100 * correct / total:.2f}%")


        # Check for early stopping, if 10 continuous increments, then early stop
        if current_eval_loss < best_val_loss:
            best_val_loss = current_eval_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered! at epoch:{epoch}")
                break

    # plot train and validation loss
    plt.plot(train_losses, label="train")
    plt.plot(eval_losses, label="eval")
    plt.legend()
    plt.show()


    model.eval()
    all_scores = []
    all_preds = []
    all_true = []

    # testing
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cpu'), target.to('cpu')
            output = model(data)

            # Get predicted class
            pred = output.argmax(dim=1, keepdim=True).squeeze()

            all_scores.extend(output.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_true.extend(target.cpu().numpy())

    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Accuracy
    accuracy = accuracy_score(all_true, all_preds)

    # One-hot encode true labels for macro ROC and PR AUC
    all_true_bin = label_binarize(all_true, classes=np.unique(all_true))

    # ROC AUC
    roc_auc = roc_auc_score(all_true_bin, all_scores, average='macro', multi_class='ovr')

    # PR AUC
    pr_score = average_precision_score(all_true_bin, all_scores, average='macro')

    # F1 Score
    f1 = f1_score(all_true, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Macro ROC AUC: {roc_auc:.2f}")
    print(f"Macro PR AUC: {pr_score:.2f}")
    print(f"Macro F1 Score: {f1:.2f}")

    # Micro-averaged Precision, Recall, and F1 Score
    micro_precision = precision_score(all_true, all_preds, average='micro')
    micro_recall = recall_score(all_true, all_preds, average='micro')
    micro_f1 = f1_score(all_true, all_preds, average='micro')

    # Micro-averaged ROC AUC
    micro_roc_auc = roc_auc_score(all_true_bin.ravel(), all_scores.ravel())

    print(f"Micro-averaged Precision: {micro_precision:.2f}")
    print(f"Micro-averaged Recall: {micro_recall:.2f}")
    print(f"Micro-averaged F1 Score: {micro_f1:.2f}")
    print(f"Micro-averaged ROC AUC: {micro_roc_auc:.2f}")


    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=np.unique(all_true), yticklabels=np.unique(all_true))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # save model
    torch.save(model.state_dict(), f'nn_component_identification.pth')


def execute(model):
    """
    main process
    :param model: model type
    :return: None
    """
    def file_reader():
        ref_tags = []
        with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/train.txt') as file:
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
    x, y = transform(data, ner=False)
    print(x, y)
    if model != 'nn':
        best_model = train(x, y, model_type=model)

        # store data
        with open(os.path.dirname(os.getcwd()) + '/' + f'component_identification/{model}_component_identification.pkl', 'wb') as file:
            pickle.dump(best_model, file)
    else:
        train_nn(x, y)


if __name__ == '__main__':
    execute(model='nn')
