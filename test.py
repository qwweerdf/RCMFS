import pypandoc


# ['asciidoc', 'asciidoctor', 'beamer', 'biblatex', 'bibtex', 'commonmark', 'commonmark_x', 'context', 'csljson', 'docbook', 'docbook4', 'docbook5', 'docx', 'dokuwiki', 'dzslides', 'epub', 'epub2', 'epub3', 'fb2', 'gfm', 'haddock', 'html', 'html4', 'html5', 'icml', 'ipynb', 'jats', 'jats_archiving', 'jats_articleauthoring', 'jats_publishing', 'jira', 'json', 'latex', 'man', 'markdown', 'markdown_github', 'markdown_mmd', 'markdown_phpextra', 'markdown_strict', 'mediawiki', 'ms', 'muse', 'native', 'odt', 'opendocument', 'opml', 'org', 'pdf', 'plain', 'pptx', 'revealjs', 'rst', 'rtf', 's5', 'slideous', 'slidy', 'tei', 'texinfo', 'textile', 'xwiki', 'zimwiki']
if __name__ == '__main__':
    # Convert the Word document to XHTML
    output = pypandoc.convert_file('/Users/jialong/Desktop/CSC3064-Report.odt', '')

    print(output)


