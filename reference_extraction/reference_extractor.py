import sys
import argparse
import re
import linecache
import os
import pypandoc

from bs4 import BeautifulSoup

"""
rule based reference extraction algorithms
"""

REF_LIST = [
    'reference',
    'references',
    'bibliography'
]


def reference_extraction_odt_docx(path):
    """
    extract references from docx and dot
    :param path: path of the documents
    :return: reference list
    """

    extension = os.path.splitext(path)[1][1:]
    print(extension)
    html = pypandoc.convert_file(path, 'html5')
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    references = []
    is_ref = False
    ref_tag_name = ''

    for tag in soup:

        # check if matches with the regex
        if re.match(r'^\s*(references?|bibliography)\s*$', tag.text.lower()):
            # the next item is a reference, so append the next one into references list if the tag name is ol or ul
            ref_tag_name = tag.find_next_sibling().name
            if extension == 'odt' and ref_tag_name == ('ol' or 'ul'):
                references.append(tag.find_next_sibling().text.replace('\n', ''))
            is_ref = True
            continue

        if is_ref:
            # if there is none tab format then continue
            if tag.name is None:
                continue

            # if the html file is in p format
            if ref_tag_name == 'p':
                if tag.name is not None and tag.name != ref_tag_name:
                    break
                references.append(tag.text.replace('\n', ''))

            # if the html file is in the list format
            if ref_tag_name == 'ol' or ref_tag_name == 'ul':
                if extension == 'odt':
                    if tag.find_next_sibling() is None:
                        break
                    references.append(tag.find_next_sibling().text.replace('\n', ''))
                else:
                    references = tag.text.split('\n')[1:-1]
                    break

    for ref in references:
        print(ref)
    return references


def reference_extraction_tex(path):
    """
    extract references from tex file
    :param path: path of the documents
    :return: reference list
    """

    is_ref = False
    ref_content = []

    # open tex file
    with open(path, 'r') as f:
        current_ref = ''
        # iterate the file line by line
        for i, line in enumerate(f):
            # strip the line
            strip_line = line.strip()
            # find the beginning of the bibliography
            if strip_line.find(r'\begin{thebibliography}') != -1:
                is_ref = True
                continue
            # ending condition
            if strip_line.find(r'\end{thebibliography}') != -1:
                ref_content.append(current_ref)
                break
            # append lines within the bibliography
            if is_ref:
                # check if the next line is the new bib item or not
                if re.match(r'\\bibitem{.*}', linecache.getline(path, i+1).strip()):
                    ref_content.append(current_ref)
                    current_ref = ''
                    continue
                # append to current reference
                current_ref += strip_line

    for ref in ref_content:
        print(ref)
    return ref_content[1:]


def reference_writer(reference_list):
    with open('../test/reference_list.txt', 'w') as f:
        for item in reference_list:
            f.write('%s\n' % item)


def extract_reference(path=None):
    """
    main function to extract files
    :param path: file path
    :return: reference list
    """

    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, help='file path, including the file name')

        args = parser.parse_args()
        path = args.path
        if path is None:
            print('please specify the file path')
            sys.exit()

    file_type = path.split('.')[-1]

    # check file types
    if file_type == 'odt' or file_type == 'docx':
        reference_list = reference_extraction_odt_docx(path)
        reference_writer(reference_list)
        return reference_list
    elif file_type == 'tex':
        reference_list = reference_extraction_tex(path)
        reference_writer(reference_list)
        return reference_list
    elif file_type == 'pdf':
        # reference_extraction_pdf(path)
        pass
    else:
        print('file type not supported')
        sys.exit()



if __name__ == '__main__':
    extract_reference()
