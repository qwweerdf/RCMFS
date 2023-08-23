import os
from reference_extraction.reference_extractor_ml import *
from component_identification.component_identifier import *
from feedback_grading.grading import grade

if __name__ == '__main__':

    # path = '/Users/jialong/Desktop/ref.bib'
    # user defined variables
    # model_type: svm/brf/nn/perceptron
    path = '/Users/jialong/PycharmProjects/RCMFS/test_docs/t3.docx'
    model_type = 'nn'



    extension = os.path.splitext(path)[1][1:]
    if extension == 'doc' or extension == 'docx':
        extract_ref(path, model_type=model_type)
    print(extension)
    components = get_components(ftype=extension, model_type='nn')
    print(components)
    grades, fb = grade()
    for line in fb:
        print(line)