import pickle
import sys

import torch

from reference_extraction import ref_ext
import pandas as pd
import numpy as np
from docx import Document
import os
import xgboost as xgb


def extract_features(para):
    # form to DataFrame
    data = pd.DataFrame(para, columns=['content'])
    # extract features
    feats = np.array(list(ref_ext.feature_extraction(data)))
    # do transpose
    return list(map(list, zip(*(feats.tolist()))))


def extract_ref(file_path, model_type='svm'):
    if model_type != 'nn':
        with open(os.path.dirname(os.getcwd()) + '/' + f'reference_extraction/{model_type}_reference_extraction.pkl', 'rb') as file:
            model = pickle.load(file)


    def read_word_doc(file_path):
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return text


    # Example usage
    document_text = read_word_doc(file_path)

    paragraphs = []

    # data cleaning and append document to a list
    for paragraph in document_text:
        if paragraph == "" or paragraph == '\n' or paragraph == '\t':
            continue
        paragraphs.append(paragraph)
    paragraphs = np.array(paragraphs)

    # feature extraction
    data = extract_features(paragraphs)

    if model_type == 'nn':
        nn_model = ref_ext.SimpleNN(len(data[0]))
        nn_model_path = os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/nn_reference_extraction.pth'
        nn_model.load_state_dict(torch.load(nn_model_path))
        nn_model.eval()  # Set the model to evaluation mode
        output = nn_model(torch.tensor(data).float())
        pred = [0 if item < 0.5 else 1 for item in output.detach().numpy()]
    else:
        # predict paragraphs
        pred = model.predict(data)

    print(pred)
    refs = []
    for i in range(paragraphs.shape[0]):
        if pred[i] == 1:
            refs.append(paragraphs[i])

    with open(os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/extracted_references.txt', 'w') as file:
        for row in refs:
            row = row.replace('\u00A0', ' ')
            file.write(row + '\n')
    for ref in refs:
        print(ref)
    return refs

    # test = pd.DataFrame([
    #                         '5.	Xu, X., Xu, A., Jiang, Y., Wang, Z., Wang, Q., Zhang, Y. and Wen, H., 2020, November. Research on Security Issues of Docker and Container Monitoring System in Edge Computing System. In Journal of Physics: Conference Series (Vol. 1673, No. 1, p. 012067). IOP Publishing.']
    #
    #                     , columns=['content'])
    #
    # print(ref_ext.feature_extraction(test))
    # print(model.predict([[1, 3, 0, 3, 0, 0, 4, 0]]))

    # chunk_size = 10
    # for i in range(0, data.shape[0], chunk_size):
    #     part = data['content'][i:i+chunk_size]
    #     feat_list = list(ref_ext.feature_extraction(part))
    #     print(feat_list)

    # ref_ext.feature_extraction()


if __name__ == '__main__':
    print(extract_ref('example path'))
