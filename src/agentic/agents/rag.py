from ..models.generation_model import GenerationModel
from ..models.embedding_model import EmbeddingModel
import numpy as np
import re
from collections import Counter
from nltk.stem import PorterStemmer  # For stemming


class RAG:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        documents: list[str] = [],
        selection: str = "emb",
    ):
        self.emb_model = embedding_model
        self.documents = documents  # list of tuples (title, content)
        if selection == "freq":
            self.stemmer = PorterStemmer()
            self.idf = {}
            self._compute_idf()

        else:  # the else case is just : =="emb"
            self.emb_titles = []
            self._compute_emb_titles()

    def similarity(self, vec1: list, vec2: list) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def corresponding_documents_emb(self, question: str):
        question_emb = self.emb_model.generate_embedding(question)

        n = len(self.documents)
        infor_doc = ""
        scores = []
        indices = [j for j in range(n)]
        for i in range(n):
            emb = self.emb_titles[i]
            score = self.similarity(question_emb, emb)
            scores.append(score)
        indices.sort(reverse=True, key=lambda i: scores[i])

        for k in range(3):  # only the top 3 documents
            infor_doc += self.documents[indices[k]][1] + "\n\n"

        print("selected documents (embedding-based) : ", indices[:3])
        return infor_doc

    def _compute_emb_titles(self):

        title_embeddings = self.emb_model.generate_embeddings(
            [doc[0] for doc in self.documents]
        )
        self.emb_titles.extend(title_embeddings)

    def stem_custom(self, word):
        if word.isupper() and len(word) >= 3:  # Preserve acronyms like MCMC
            return word.lower()
        return self.stemmer.stem(word)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extracts meaningful keywords (removes stopwords)."""
        words = re.findall(r"\w+", text.lower())

        # Basic stopwords list (expand as needed)
        stop_set = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "in",
            "to",
        }

        return [word for word in words if word not in stop_set]

    def corresponding_documents_freq(self, question: str):
        # Step 1: Extract stemmed keywords (remove stopwords)
        keywords = [self.stem_custom(word) for word in self._extract_keywords(question)]

        if not keywords:
            print("no keywords found ! ")
            return

        # Step 2: Rank documents by keyword overlap
        scores = []
        for i, (title, content) in enumerate(self.documents):

            content_words = [
                self.stem_custom(w) for w in re.findall(r"\w+", content.lower())
            ]
            content_word_counts = Counter(content_words)

            score = sum(
                (content_word_counts[word] / len(content_words))
                * self.idf.get(word, 1.0)
                for word in keywords
                if word in content_word_counts
            )
            scores.append((score, i))

        scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for (score, idx) in scores[:3]]

        print("Selected documents (keyword-based):", top_indices)

        return "\n\n".join(self.documents[idx][1] for idx in top_indices)

    def _compute_idf(self):
        """Precompute IDF with stemmed terms."""
        doc_count = len(self.documents)

        all_terms = []
        for title, content in self.documents:

            content_terms = [
                self.stem_custom(word) for word in re.findall(r"\w+", content.lower())
            ]
            all_terms.extend(content_terms)

        unique_terms = set(all_terms)

        for term in unique_terms:
            docs_with_term = sum(
                1
                for title, content in self.documents
                if term
                in [self.stem_custom(w) for w in re.findall(r"\w+", content.lower())]
            )
            self.idf[term] = np.log(doc_count / (1 + docs_with_term))