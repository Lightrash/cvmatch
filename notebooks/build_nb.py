import nbformat as nbf, sys
def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
    for kind, src in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind=="md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path)
