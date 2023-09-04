import subprocess

import fuzzywuzzy.fuzz
import bibtexparser
import json
import component_identification.component_identification
from online_datasource.online_reference_extractor import *
from component_identification.component_identification import *

"""
main process to do component identification for the uploaded files
"""


def preprocess(ref_str):
    """
    preprocess reference string
    :param ref_str: reference string
    :return: new reference string list
    """
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


def get_components(ftype, model_type='svm', ner=True):
    """
    get components from model prediction
    :param ftype: file type
    :param model_type: model type
    :param ner: if Flair NER
    :return: reference dictionary
    """

    # if BibTeX, parse the file and compare with online datasource
    if ftype == 'bib':
        bib_ref_compare = []
        with open('/Users/jialong/Desktop/ref.bib', 'r') as f:
            bib_database = bibtexparser.load(f)
        for each in bib_database.entries:
            bib_ref_dict = {}
            if 'author' in each:
                bib_ref_dict['authors'] = each['author']
            if 'title' in each:
                bib_ref_dict['title'] = each['title']
            if 'volume' in each:
                bib_ref_dict['volume'] = each['volume']
            if 'issue' in each:
                bib_ref_dict['issue'] = each['issue']
            if 'journal' in each:
                bib_ref_dict['journal'] = each['journal']
            if 'pages' in each:
                bib_ref_dict['pages'] = each['pages']
            if 'year' in each:
                bib_ref_dict['year'] = each['year']
            if 'doi' in each:
                bib_ref_dict['doi'] = each['doi']
            bib_online_ref = pubmed(bib_ref_dict['title'])
            print(bib_ref_dict)
            print(bib_online_ref)
            if bib_online_ref != "":
                bib_ref_compare.append(bib_ref_dict)
                bib_ref_compare.append(bib_online_ref)
        with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt', 'w') as f:
            for item in bib_ref_compare:
                f.write(f'{item}\n')
        return ""

    ref_list = []
    ref_dict = {}
    ref_compare = []
    with open(os.path.dirname(os.getcwd()) + '/config.json', 'r') as f:
        config = json.load(f)
    with open(os.path.dirname(os.getcwd()) + '/' + "reference_extraction/extracted_references.txt", "r") as file:
        # Read the contents of the file
        counter = 0
        for line in file:

            # Execute the CERMINE, the software is not included in the project because the limitation of uploaded zip size (30MB)
            # if you need that, please download it here: https://maven.ceon.pl/artifactory/kdd-releases/pl/edu/icm/cermine/cermine-impl/1.13/cermine-impl-1.13-jar-with-dependencies.jar
            cermine_path = config["cermine_path"]
            result = subprocess.run(['java', '-cp', cermine_path, 'pl.edu.icm.cermine.bibref.CRFBibReferenceParser', '-reference', line.strip()], capture_output=True, text=True)

            ref = result.stdout
            if ref[9:16] == 'Unknown':
                continue

            counter += 1
            ref_list.append(preprocess(ref))
        print(f"extracted reference count: {counter}")

        # load models
        if model_type != 'nn':
            with open(os.path.dirname(os.getcwd()) + '/' + f'component_identification/{model_type}_component_identification.pkl', 'rb') as file:
                model = pickle.load(file)
        else:
            nn_model = component_identification.component_identification.TabularNN(8, 50, 8)
            nn_model_path = os.path.dirname(os.getcwd()) + '/' + 'component_identification/nn_component_identification.pth'
            nn_model.load_state_dict(torch.load(nn_model_path))
            nn_model.eval()  # Set the model to evaluation mode


        # main loop to compare references
        for ref in ref_list:
            ref_dict = {}
            data = pd.DataFrame(ref, columns=['content'])
            feats = np.array(list(feature_extraction(data, ner=ner)))
            # do transpose
            x = list(map(list, zip(*(feats.tolist()))))
            if model_type != 'nn':
                res = model.predict(x)
                print(res)
            else:
                output = nn_model(torch.tensor(x).float())
                res = []
                for each in output:
                    res.append(torch.argmax(each).numpy().item())

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
            try:
                online_ref = pubmed(ref_dict['title'])
            except KeyError:
                continue
            print(online_ref)
            try:
                if fuzzywuzzy.fuzz.token_set_ratio(ref_dict['title'], online_ref['title']) <= 67 or fuzzywuzzy.fuzz.token_set_ratio(ref_dict['authors'], online_ref['authors']) <= 50:
                    print('too away!!')
                    continue
            except KeyError:
                continue
            print()
            if online_ref != "":
                ref_compare.append(ref_dict)
                ref_compare.append(online_ref)
    with open(os.path.dirname(os.getcwd()) + '/' + 'component_identification/ref_compare.txt', 'w') as f:
        for item in ref_compare:
            f.write(f'{item}\n')
    return ref_dict


if __name__ == '__main__':
    get_components(ftype='docx')