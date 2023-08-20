from reference_extraction.reference_extractor_ml import *
from component_identification.component_identifier import *
from feedback_grading.grading import grade
import os

if __name__ == '__main__':
    """
    example documents:
    
    /Users/jialong/Library/CloudStorage/OneDrive-Personal/CSC3064-Report.docx
    /Users/jialong/Library/CloudStorage/OneDrive-Personal/dissertation/JialongZhuDissertation.docx
    
    /Users/jialong/Desktop/CSC3064-Report.odt
    /Users/jialong/Desktop/CSC3065Assignment1.odt
    
    /Users/jialong/Desktop/main.tex
    """
    # rule-based
    # reference_extractor.extract_reference('/Users/jialong/Desktop/CSC3065Assignment1.pdf')

    # machine learning based
    """
    Pipline:
    """
    # path = '/Users/jialong/Desktop/ref.bib'
    # extension = os.path.splitext(path)[1][1:]
    # if extension == 'doc' or extension == 'docx':
    #     extract_ref(path)
    # print(extension)
    # components = get_components(ftype=extension)
    # print(components)
    # grade()



    """
    example titles:
    
    Deep learning for chest X-ray analysis: A survey.
    The Emergence of Natural Language Quantification.
    Effects of Language on Visual Perception.
    """

    # print(online_reference_extractor.pubmed('Deep learning for chest X-ray analysis: A survey.'))