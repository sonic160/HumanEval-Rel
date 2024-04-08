import pytest
from model import HuggingFace

@pytest.fixture(scope="module")
def huggingface_model():
    # Initialise le modèle HuggingFace une fois avant tous les tests du module
    return HuggingFace('Croissant', 'croissantllm/CroissantLLMBase')

def test_model_init(huggingface_model):
    # Teste si le modèle est correctement initialisé
    huggingface_model.model_init()
    assert (huggingface_model._HuggingFace__tokenizer, huggingface_model._HuggingFace__model) != (None, None)

def test_generate(huggingface_model):
    # Teste la génération de texte à partir d'un prompt
    prompt = '''I am so tired I could sleep right now. -> Je suis si fatigué que je pourrais m'endormir maintenant.\n
            He is heading to the market. -> Il va au marché.\n
            We are running on the beach. -> '''
    result = huggingface_model.generate(prompt)
    assert result is None  # Vérifie que le résultat est None après la génération car il n'y a pas de code
