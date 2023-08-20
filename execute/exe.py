import os
from reference_extraction.reference_extractor_ml import *
from component_identification.component_identifier import *
from feedback_grading.grading import grade

if __name__ == '__main__':

    path = '/Users/jialong/Desktop/ref.bib'
    extension = os.path.splitext(path)[1][1:]
    if extension == 'doc' or extension == 'docx':
        extract_ref(path)
    print(extension)
    components = get_components(ftype=extension)
    print(components)
    grades, fb = grade()
    for line in fb:
        print(line)