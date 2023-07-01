import pickle
import ref_ext
import pandas as pd
import numpy as np
from docx import Document


def extract_features(para):
    # form to DataFrame
    data = pd.DataFrame(para, columns=['content'])
    # extract features
    feats = np.array(list(ref_ext.feature_extraction(data)))
    # do transpose
    return list(map(list, zip(*(feats.tolist()))))


if __name__ == '__main__':

    with open('svm_model.pkl', 'rb') as file:
        model = pickle.load(file)


    def read_word_doc(file_path):
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return text


    # Example usage
    # file_path = '/Users/jialong/Downloads/CSC3065 Assignment 1.docx'
    file_path = '/Users/jialong/PycharmProjects/RCMFS/files/t1.docx'
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

    # predict paragraphs
    pred = model.predict(data)


    for i in range(paragraphs.shape[0]):
        print(paragraphs[i])
        print(pred[i])
        print()
        print()
    print('ref:')
    print(np.count_nonzero(pred == 1))
    print('non ref:')
    print(np.count_nonzero(pred == 0))

    for i in range(len(data)):
        print(data[i])
        print(pred[i])

    for i in range(paragraphs.shape[0]):
        if pred[i] == 1:
            print(paragraphs[i])

