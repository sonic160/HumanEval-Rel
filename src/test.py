import pytest
import json
from model import HuggingFace
from benchmarker import Benchmarker

@pytest.fixture(scope="module")

def huggingface_model():
    # Initialise le modèle HuggingFace une fois avant tous les tests du module
    return HuggingFace('Croissant', 'croissantllm/CroissantLLMBase')

@pytest.fixture
def challenges_data():
    return [
        {"task_id": 1, "prompt": "Prompt 1", "test": "Test 1", "entry_point": "Entry 1"},
        {"task_id": 2, "prompt": "Prompt 2", "test": "Test 2", "entry_point": "Entry 2"},
    ]

@pytest.fixture
def completions_data():
    return [
        {"task_id": 1, "completion": "Completion 1"},
        {"task_id": 2, "completion": "Completion 2"},
    ]

def test_model_init(huggingface_model):
    # Teste si le modèle est correctement initialisé
    huggingface_model.model_init()
    assert (huggingface_model._HuggingFace__tokenizer, huggingface_model._HuggingFace__model) != (None, None)

def test_generate(huggingface_model):
    # Teste la génération de texte à partir d'un prompt. Si il n'y a pas de code dans le prompt le string renvoyé est vide

    prompt1 = '''I am so tired I could sleep right now. -> Je suis si fatigué que je pourrais m'endormir maintenant.\n
            He is heading to the market. -> Il va au marché.\n
            We are running on the beach. -> '''
    
    
    result1 = huggingface_model.generate(prompt1)
    assert (result1 == "")  # Vérifie que le résultat est None après la génération car il n'y a pas de code
    
    prompt2 = '''Write a code that can calculate a matrix dot product'''
    result2 = huggingface_model.generate(prompt2)
    assert  (result2 != "")
    assert isinstance(result2, str)

def test_extract(huggingface_model):
    #Teste l'extraction de code d'un string

    str1 = '''Sure, here's a Python function that calculates the square of a matrix:

    python
    def square_matrix(matrix):
        """
        Calculates the square of a matrix.

        Args:
            matrix (list): A list of lists representing a matrix.

        Returns:
            int: The square of the matrix.
        """

        #begin code

        result = 0
        for row in matrix:
            result += sum(row)
        return result


    You can use this function to calculate the square of a matrix by passing it as a list of lists. For example, if you have a matrix [1, 2, 3, 4, 5], you can calculate its square by calling square_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]).square(). The function will return the square of the matrix, which in this case is 25.'''

    result1 = huggingface_model.extract(str1)
    assert (result1 == '''def square_matrix(matrix):\n  \n  result = 0\n    for row in matrix:\n        result += sum(row)\n    return result''')

    str2 = '''Voici un exemple de fonction Python qui n'a pas de déclaration return à la fin :

    python

    def afficher_message():
        print("Bonjour !")
        # Pas de return statement ici

    # Appel de la fonction
    afficher_message()'''

    result2 = huggingface_model.extract(str2)
    assert (result2 == '''def afficher_message():\n    print("Bonjour !")''')

