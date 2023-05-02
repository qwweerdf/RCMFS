import requests
import json


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
    summ = summary_json['result'][ids[0]]



    # generate the customised dict
    pubmed_ref = {}

    authors = []
    for author in summ['authors']:
        authors.append(author['name'])
    pubmed_ref['authors'] = authors

    pubmed_ref['title'] = summ['title']

    pubmed_ref['volume'] = summ['volume']

    pubmed_ref['issue'] = summ['issue']

    pubmed_ref['pages'] = summ['pages']

    for idtype in summ['articleids']:
        if idtype['idtype'] == 'doi':
            pubmed_ref['doi'] = idtype['value']

    return pubmed_ref
