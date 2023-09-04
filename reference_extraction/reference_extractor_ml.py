import pickle
import torch

from reference_extraction import ref_ext
import pandas as pd
import numpy as np
from docx import Document
import os

"""
machine learning based reference extraction for uploaded files
"""


def extract_features(para):
    """
    give paragraphs, converts to features
    :param para: paragraphs
    :return: list of features
    """

    # form to DataFrame
    data = pd.DataFrame(para, columns=['content'])
    # extract features
    feats = np.array(list(ref_ext.feature_extraction(data)))
    # do transpose
    return list(map(list, zip(*(feats.tolist()))))


def extract_ref(file_path, model_type='svm'):
    """
    extract reference main llop
    :param file_path
    :param model_type
    :return: reference list
    """

    if model_type != 'nn':
        with open(os.path.dirname(os.getcwd()) + '/' + f'reference_extraction/{model_type}_reference_extraction.pkl',
                  'rb') as file:
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
        if paragraph == "" or paragraph == '\n' or paragraph == '\t' or '\n' in paragraph or '\t' in paragraph or paragraph.isspace():
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
    print(len(refs))
    return refs


if __name__ == '__main__':
    print(extract_ref('example path'))
