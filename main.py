import reference_extractor
import online_reference_extractor
from reference_extractor_ml import *
from component_identifier import *

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
    extract_ref('files/t1.docx')
    components = get_components()
    print(components)




    """
    example titles:
    
    Deep learning for chest X-ray analysis: A survey.
    The Emergence of Natural Language Quantification.
    Effects of Language on Visual Perception.
    """

    # print(online_reference_extractor.pubmed('Deep learning for chest X-ray analysis: A survey.'))