import requests
import json
import time


def pubmed(title):
    # get searched ids
    query_url = r'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="{}"[Title:~{}]&retmax=3&retmode=json'.format(
        title, len(title.split()))
    id_response = requests.get(query_url).text

    id_json = json.loads(id_response)
    ids = id_json['esearchresult']['idlist']

    # get the summary for the returned articles
    summary_url = r'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}&retmode=json'.format(
        ','.join(ids))

    summary_response = requests.get(summary_url).text
    summary_json = json.loads(summary_response)

    try:
        summ = summary_json['result'][ids[0]]
    except KeyError:
        print('cannot find the search result in pubmed, try search at crossref!')
        cross_ref_res = crossref(title)
        if cross_ref_res != "":
            return cross_ref_res
        return ""

    # generate the customised dict
    pubmed_ref = {}

    authors = []
    for author in summ['authors']:
        authors.append(author['name'])
    pubmed_ref['authors'] = authors

    pubmed_ref['title'] = summ['sorttitle']

    pubmed_ref['volume'] = summ['volume']

    pubmed_ref['issue'] = summ['issue']

    pubmed_ref['pages'] = summ['pages']

    pubmed_ref['journal'] = summ['source']

    pubmed_ref['year'] = summ['pubdate'][:4]

    for idtype in summ['articleids']:
        if idtype['idtype'] == 'doi':
            pubmed_ref['doi'] = idtype['value']

    return pubmed_ref




def crossref(title):
    res = requests.get(
        f'https://api.crossref.org/works?query={title}&rows=1')

    original_ref_dict = res.json()
    try:
        summ = original_ref_dict['message']['items'][0]
    except KeyError:
        print('cannot find the search result in crossref, continue to the next one!')
        return ""


    ref_dict = {}

    try:
        authors = []
        for author in summ['author']:
            full_name = author['given'] + " " + author['family']
            authors.append(full_name)

        ref_dict['authors'] = authors
    except KeyError:
        pass


    try:
        ref_dict['title'] = summ['title'][0]
    except KeyError:
        pass

    try:
        ref_dict['volume'] = summ['volume']
    except KeyError:
        pass

    try:
        ref_dict['issue'] = summ['journal-issue']['issue']
    except KeyError:
        pass

    try:
        ref_dict['pages'] = summ['page']
    except KeyError:
        pass

    try:
        ref_dict['journal'] = summ['short-container-title'][0]
    except KeyError:
        pass

    try:
        ref_dict['year'] = summ['published']['date-parts'][0][0]
    except KeyError:
        pass

    try:
        ref_dict['doi'] = summ['DOI']
    except KeyError:
        pass

    return ref_dict



def get_ref(title):
    curr_ref = []
    res = pubmed(str(title))
    curr_ref.append((', '.join(res['authors']), 'authors'))
    curr_ref.append((res['title'], 'title'))
    curr_ref.append((res['volume'], 'volume'))
    curr_ref.append((res['issue'], 'issue'))
    curr_ref.append((res['pages'], 'pages'))
    curr_ref.append((res['journal'], 'journal'))
    curr_ref.append((res['year'], 'year'))
    try:
        curr_ref.append((res['doi'], 'doi'))
    except:
        curr_ref.append(('', 'doi'))
    return curr_ref


if __name__ == '__main__':
    ref_tags = []
    for i in range(201, 601):
        curr_ref = []
        res = pubmed(str(i))
        curr_ref.append((', '.join(res['authors']), 'authors'))
        curr_ref.append((res['title'], 'title'))
        curr_ref.append((res['volume'], 'volume'))
        curr_ref.append((res['issue'], 'issue'))
        curr_ref.append((res['pages'], 'pages'))
        curr_ref.append((res['journal'], 'journal'))
        curr_ref.append((res['year'], 'year'))
        try:
            curr_ref.append((res['doi'], 'doi'))
        except:
            curr_ref.append(('', 'doi'))
        ref_tags.append(curr_ref)
        time.sleep(1)
        print(i)

    with open('component_identification/train.txt', 'a') as file:
        # Write each element of the list on a separate line
        for row in ref_tags:
            line = repr(row)
            file.write(line + '\n')