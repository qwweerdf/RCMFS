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

    print(summary_json)
    try:
        summ = summary_json['result'][ids[0]]
    except KeyError:
        print('cannot find the search result, continue to the next one!')
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
    for i in range(200):
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

    with open('train.txt', 'a') as file:
        # Write each element of the list on a separate line
        for row in ref_tags:
            line = repr(row)
            file.write(line + '\n')