import os


def load_documents(path) -> list[tuple[str, str]]:
    """
    Loads and parses documents from a text file, returning a list of (title, content) tuples.

    Each document in the file is assumed to start with a single '#' heading line (not beginning 
    with '##' and not equal to '# Python verification'). The content of the document includes the 
    heading line and all subsequent lines until the next heading line is encountered.

    Args:
        path (str): The path to the text file containing the documents.

    Returns:
        list[tuple[str, str]]: A list of tuples where each tuple contains:
            - title (str): The title of the document (extracted from the heading line).
            - content (str): The full content of the document including the heading line.
    """

    documents = []  # list of tuples (title, content)

    with open(path, "r") as database_file:
        title, doc = "", ""
        for line in database_file:
            if (
                line[0] == "#"
                and line[1] != "#"
                and not line.startswith("# Python verification")
            ):  # on arrive à un nouveau document
                documents.append((title, doc))  # terminer le doc précédent
                title = line[1:-1]
                doc = line

            else:
                doc += line

    documents.append((title, doc))

    documents.pop(0)

    return documents
