import os
from reference_extraction.reference_extractor_ml import *
from component_identification.component_identifier import *
from feedback_grading.grading import grade
import time
import sys

"""
testing main function
"""

if __name__ == '__main__':

    # path = '/Users/jialong/Desktop/ref.bib'
    # user defined variables
    # model_type: svm/brf/nn/perceptron
    model_list = ['nn', 'brf', 'perceptron', 'svm']
    path = '/Users/jialong/PycharmProjects/RCMFS/test_docs/t7.docx'
    model_type = model_list[3]


    # DO PLIE-LINE
    extension = os.path.splitext(path)[1][1:]
    if extension == 'doc' or extension == 'docx':
        start = time.time()
        refs = extract_ref(path, model_type=model_type)
        end = time.time()
        print(f'time consumed for reference extraction: {(end - start)}')
        if len(refs) == 0:
            sys.exit('no reference extracted')
    print(extension)
    start = time.time()
    components = get_components(ftype=extension, model_type=model_list[0], ner=True)
    end = time.time()
    print(f"time consumed for component identification: {(end - start)}")

    print(components)
    grades, fb, final_summ = grade()
    for line in fb:
        print(line)
    for line in final_summ:
        print(line)

    extracted_references = os.path.dirname(os.getcwd()) + '/' + 'reference_extraction/extracted_references.txt'
    ref_compare = os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt'

    # delete temp files
    if os.path.exists(extracted_references):
        os.remove(extracted_references)
        print(f"'{extracted_references}' has been deleted!")
    else:
        print(f"'{extracted_references}' does not exist!")
    if os.path.exists(ref_compare):
        os.remove(ref_compare)
        print(f"'{ref_compare}' has been deleted!")
    else:
        print(f"'{ref_compare}' does not exist!")
