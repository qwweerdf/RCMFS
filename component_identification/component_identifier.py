import pickle
import re
import subprocess
import os
import numpy as np
import pandas as pd

from online_datasource.online_reference_extractor import *
import time
from component_identification.component_identification import *


def preprocess(ref_str):
    ref_str_list = ref_str.split('\n')
    new_ref_str_list = []
    for i, item in enumerate(ref_str_list):
        if i != 0 and item != '}' and i != len(ref_str_list)-1:
            item = item.replace('\t', '')
            if item.split(' ')[0] == 'title':
                item = item.lower()
                new_ref_str_list.append(re.findall(r'\{.*?}', item)[0][1:-1])
                continue
            if item.split(' ')[0] == 'pages':
                pages = re.findall(r'\{.*?}', item)[0][1:-1]
                if pages.find('--') != -1:
                    pages = pages.replace("-", "", 1)
                new_ref_str_list.append(pages)
                continue
            print(item)
            values = re.findall(r'\{.*?}', item)[0][1:-1]
            new_ref_str_list.append(values)
    print()
    return new_ref_str_list


def get_components():
    # resp = requests.post('http://cermine.ceon.pl/parse.do', data={
    #     'reference': "Ansari, U. B., & Sarode, T. (2017). Skin cancer detection using image processing. Int. Res. J. Eng. Technol, 4(4), 2875-2881."})
    # print(resp.content)
    # data = pd.DataFrame(['November 2022'], columns=['content'])
    # with open('svm_component_identification.pkl', 'rb') as file:
    #     model = pickle.load(file)
    # feats = np.array(list(feature_extraction(data)))
    # print(feats)
    # # do transpose
    # x = list(map(list, zip(*(feats.tolist()))))
    # print(model.predict(x))



    ref_list = []
    ref_dict = {}
    ref_compare = []
    with open(os.path.dirname(os.getcwd()) + '/' + "reference_extraction/extracted_references.txt", "r") as file:
        # Read the contents of the file
        for line in file:

            # Execute a command
            result = subprocess.run(['java', '-cp', '/Users/jialong/Downloads/cermine.jar', 'pl.edu.icm.cermine.bibref.CRFBibReferenceParser', '-reference', line.strip()], capture_output=True, text=True)

            # Print the command output
            ref = result.stdout
            if ref[9:16] == 'Unknown':
                continue
            # matches = re.findall(r'{(.*?)}', ref)
            # ref_list.append(matches)
            ref_list.append(preprocess(ref))


        with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/svm_component_identification.pkl', 'rb') as file:
            model = pickle.load(file)

        for ref in ref_list:
            ref_dict = {}
            data = pd.DataFrame(ref, columns=['content'])
            feats = np.array(list(feature_extraction(data, ner=True)))
            # do transpose
            x = list(map(list, zip(*(feats.tolist()))))
            res = model.predict(x)
            for i, item in enumerate(ref):
                if res[i] == 0:
                    ref_dict['authors'] = item
                elif res[i] == 1:
                    ref_dict['title'] = item
                elif res[i] == 2:
                    ref_dict['volume'] = item
                elif res[i] == 3:
                    ref_dict['issue'] = item
                elif res[i] == 4:
                    ref_dict['pages'] = item
                elif res[i] == 5:
                    ref_dict['journal'] = item
                elif res[i] == 6:
                    ref_dict['year'] = item
                elif res[i] == 7:
                    ref_dict['doi'] = item
            print(ref_dict)
            # time.sleep(0.5)
            online_ref = pubmed(ref_dict['title'])
            print(online_ref)
            print()
            if online_ref != "":
                ref_compare.append(ref_dict)
                ref_compare.append(online_ref)
    with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt', 'w') as f:
        for item in ref_compare:
            f.write(f'{item}\n')
    return ref_dict


if __name__ == '__main__':
    get_components()