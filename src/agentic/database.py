import os

def load_documents(path):
    documents=[] # list of tuples (title, content)
    
    with open(path,"r") as database_file:
        doc, title="",""
        for line in database_file:
            if line[0]=="#" and line[1]!="#" and not line.startswith("# Python verification"):  # on arrive à un nouveau document
                documents.append((title,doc)) # terminer le doc précédent
                title=line[1:-1]
                doc=line

            else:
                doc+=line

    documents.pop(0)

    return documents


