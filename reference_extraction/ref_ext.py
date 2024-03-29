import os
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn import svm
from torch.utils.data import TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

"""
train reference extraction models
"""


# feature extraction
def feature_extraction(data):
    """
    converts text tabular data to numeric format
    :param data: text tabular data
    :return: numeric features
    """

    # flair NER, deprecated
    # tagger = Classifier.load('ner-fast')

    data_list = data.content.values.tolist()
    tokens = []
    for each in data_list:
        tokens.append(word_tokenize(each))

    # feature 1: the amount of NNP
    # feature 2: the amount of verbs
    # feature 3: comma count
    # feature 4: dot count
    # feature 5: digit count
    # feature 6: year or not
    # feature 7: amount of DT+TO+EX+CC+JJ+RB+WDT+RBR+JJR
    # feature 8: amount of MD
    # feature 9: text length
    # feature 10: if page exist (p. or pp.)
    # feature 11: if volume exist (vol.)
    feat1 = []
    feat2 = []
    feat3 = []
    feat4 = []
    feat5 = []
    feat6 = []
    feat7 = []
    feat8 = []
    feat9 = []
    feat10 = []
    feat11 = []
    for i, token in enumerate(tokens):
        tags = pos_tag(token)
        tags = np.array(tags)
        NNP_count = np.sum(tags[:, 1] == 'NNP')
        feat1.append(NNP_count)

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

        if 'p.' or 'pp' in token:
            feat10.append(1)
        else:
            feat10.append(0)

        if 'vol' in token:
            feat11.append(1)
        else:
            feat11.append(0)


    return feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11


def transform(data):
    """
    transform text tabular data to x and y
    :param data: text tabular data
    :return: features x and label y
    """

    feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11 = feature_extraction(data)
    data['nnp'] = feat1
    data['vb'] = feat2
    data['comma'] = feat3
    data['dot'] = feat4
    data['digit'] = feat5
    data['year'] = feat6
    data['sent'] = feat7
    data['md'] = feat8
    data['length'] = feat9
    data['page'] = feat10
    data['volume'] = feat11
    data.drop('content', axis=1)

    # get the mean and variance data
    for feat in ['nnp', 'vb', 'comma', 'dot', 'digit', 'year', 'sent', 'md', 'length', 'page', 'volume']:
        print('{} not reference mean: {}'.format(feat, np.mean(data[feat].loc[data['label'] == 0])))
        print('{} not reference var: {}'.format(feat, np.var(data[feat].loc[data['label'] == 0])))
        print('{} is reference mean: {}'.format(feat, np.mean(data[feat].loc[data['label'] == 1])))
        print('{} is reference var: {}'.format(feat, np.var(data[feat].loc[data['label'] == 1])))
        print()

    x = data.iloc[:, 2:].values
    y = data['label'].values

    return x, y


def train(x, y, model_type='svm'):
    """
    main training function for basic machine learning models
    :param x
    :param y
    :param model_type
    :return: the best model
    """

    accuracies = []
    balanced_accuracy_scores = []
    roc_scores = []
    pr_scores = []
    f1_scores = []
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
            model = svm.SVC(probability=True)
        elif model_type == 'perceptron':
            model = Perceptron(max_iter=1000, tol=1e-3)
        elif model_type == 'brf':
            model = RandomForestClassifier(n_estimators=100)  # 100 trees in the forest
        else:
            return "model type is not supported!"

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

        roc_score = roc_auc_score(y_test, y_pred)
        roc_scores.append(roc_score)

        pr_score = average_precision_score(y_test, y_pred)
        pr_scores.append(pr_score)

        f1_score_ = f1_score(y_test, y_pred)
        f1_scores.append(f1_score_)
        if model_type == 'perceptron':
            fpr, tpr, thres = roc_curve(y_test, model.decision_function(X_test))
        else:
            fpr, tpr, thres = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

        plt.plot(fpr, tpr, alpha=0.3)

        # if model_type == 'perceptron':
        #     p, r, thres = precision_recall_curve(y_test, model.decision_function(X_test))
        # else:
        #     p, r, thres = precision_recall_curve(y_test, model.decision_function(X_test))
        #
        # plt.plot(r, p, alpha=0.3)
        models.append(model)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('10 Folds ROC Curves')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('10 Folds PR Curves')
    plt.show()
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

    avg_cm = np.sum(cms, axis=0) / len(cms)

    sns.heatmap(np.round(avg_cm), annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    best_performance_index = pr_scores.index(max(pr_scores))

    return models[best_performance_index]


# architecture for the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_nn(X, y, params):
    """
    train neural network
    :param X
    :param y
    :param params: hyperparameters
    :return: None
    """

    # hyperparameters
    epochs = params['epochs']
    batch_size = params['batch_size']
    lr = params['lr']

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    eval_losses = []
    # Training loop
    for epoch in range(epochs):
        train_loss = 0
        for inputs, targets in train_loader:
            model.train()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, targets.reshape(-1, 1))

            loss.backward()
            optimizer.step()
            train_loss += loss

        current_train_loss = train_loss / len(train_loader)
        train_losses.append(current_train_loss.detach().numpy())
        print(f"Train Epoch {epoch + 1}/{epochs}, Loss: {current_train_loss}")

        model.eval()
        eval_loss = 0
        accuracy = 0
        for inputs, targets in test_loader:
            with torch.no_grad():
                output = model(inputs)
                loss = loss_function(output, targets.reshape(-1, 1))
                y_pred = np.where(output.numpy() < 0.5, 0, 1)
                eval_loss += loss
                accuracy += accuracy_score(targets, y_pred)
        current_eval_loss = eval_loss / len(test_loader)
        eval_losses.append(current_eval_loss.detach().numpy())
        print(f"Validation Epoch {epoch + 1}/{epochs}, Loss: {current_eval_loss}")
        print(f"Validation Epoch {epoch + 1}/{epochs}, accuracy: {accuracy / len(test_loader)}")

        # Check for early stopping, if 10 continuous increments, then early stop
        if current_eval_loss < best_val_loss:
            best_val_loss = current_eval_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered! at epoch:{epoch}")
                break

    plt.plot(train_losses, label="train")
    plt.plot(eval_losses, label="eval")
    plt.legend()
    plt.show()

    # test loop
    test_accuracies = []
    test_auc_rocs = []
    test_auc_prs = []
    test_f1_scores = []
    test_cms = []
    model.eval()
    for inputs, targets in test_loader:
        with torch.no_grad():
            output = model(inputs)
            # loss = loss_function(output, targets.reshape(-1, 1))
            y_pred = np.where(output.numpy() < 0.5, 0, 1)
            test_accuracies.append(accuracy_score(targets, y_pred))

            test_cms.append(confusion_matrix(targets, y_pred))

            probabilities = output.cpu().detach().numpy()

            fpr, tpr, _ = roc_curve(targets, probabilities)
            test_auc_rocs.append(auc(fpr, tpr))

            pr_score = average_precision_score(targets, probabilities)
            test_auc_prs.append(pr_score)

            f1_score_ = f1_score(targets, y_pred)
            test_f1_scores.append(f1_score_)

    print(f'averaged accuracies: {sum(test_accuracies) / len(test_accuracies)}')
    print(f'averaged roc score: {sum(test_auc_rocs) / len(test_auc_rocs)}')
    print(f'averaged pr score: {sum(test_auc_prs) / len(test_auc_prs)}')
    print(f'averaged f1 score: {sum(test_f1_scores) / len(test_f1_scores)}')

    avg_test_cm = np.sum(test_cms, axis=0) / len(test_cms)

    sns.heatmap(np.round(avg_test_cm), annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    torch.save(model.state_dict(), f'nn_reference_extraction.pth')
    return 0


def execute(model='svm'):
    """
    main function to train models
    :param model: model type
    :return: None
    """

    # read data
    raw = []
    with open(os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/corpus.txt') as file:
        for row in file:
            row = row.replace('\n', '')
            row = row.replace('</fnote>', '')
            row = row.replace('<fnote>', '')
            raw.append(row)

    # data preparation
    data = pd.DataFrame(raw, columns=['content'])
    data['label'] = 1
    data['label'].iloc[1350:] = 0

    # feature extraction and train
    x, y = transform(data)
    if model != 'nn':
        best_model = train(x, y, model_type=model)
        # store data
        with open(os.path.dirname(os.getcwd()) + '/' + f'reference_extraction/{model}_reference_extraction.pkl', 'wb') as file:
            pickle.dump(best_model, file)
    else:
        params = {
            'epochs': 500,
            'batch_size': 32,
            'lr': 0.0001
        }

        train_nn(x, y, params)




if __name__ == '__main__':
    execute(model='nn')
