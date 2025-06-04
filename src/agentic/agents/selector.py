from .agent import Agent
from ..models.embedding_model import EmbeddingModel
import numpy as np
import re
from collections import Counter
from nltk.stem import PorterStemmer  # For stemming

# naïf car il calcule les embeddings des documents dans chaque itération, mais on peut l optimiser facilement lmohim


class Selector(Agent):
    def __init__(self, embedding_model: EmbeddingModel, name: str="selector", documents : list[str]=[]):
        super().__init__(name)
        self.model = embedding_model
        self.documents=documents # list of tuples (title, content)
        self.stemmer = PorterStemmer()
        self.idf = {} 
        self._compute_idf()


    def _compute_idf(self):
        """Precompute IDF with stemmed terms."""
        doc_count = len(self.documents)
    
        all_terms=[]
        for title, content in self.documents:
            

            content_terms = [self.stemmer.stem(word) for word in re.findall(r'\w+', content.lower())]
            all_terms.extend(content_terms)

        unique_terms=set(all_terms)
        
        for term in unique_terms:
            docs_with_term = sum(1 for title, content in self.documents
                               if term in [self.stemmer.stem(w) for w in re.findall(r'\w+', content.lower())])
            self.idf[term] = np.log(doc_count / (1 + docs_with_term))

    def corresponding_documents_emb(self,question: str):

        question_emb=self.model.generate_embedding(question)
        documents_emb=self.model.generate_embeddings([' '.join(doc[1].split()[:25]) for doc in self.documents]) # this time we take the 1st 20 words


        n=len(self.documents)
        infor_doc=""
        scores=[]
        indices=[j for j in range(n)]
        for i in range(n):
            emb=documents_emb[i]
            score=self.similarity(question_emb,emb)
            scores.append(score)
        indices.sort(reverse=True, key= lambda i :scores[i] )


        for k in range(3):# only the top 3 documents
            infor_doc+=self.documents[indices[k]][1]+ "\n\n"

        print("selected documents : ",indices[:3])
        return infor_doc
    
    def similarity(self, vec1: list, vec2: list) -> float:

        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    

    def stem_custom(self, word):
        if word.isupper() and len(word) >= 3:  # Preserve acronyms like MCMC
            return word.lower()
        return self.stemmer.stem(word)
    



    def corresponding_documents_freq(self, question: str):
        # Step 1: Extract keywords (remove stopwords)
        keywords = [self.stem_custom(word) for word in self._extract_keywords(question)]

        if not keywords:
            print("no keywords found ! ")
            return self.corresponding_documents_emb(question)

        # Step 2: Rank documents by keyword overlap
        scores = []
        for i, (title, content) in enumerate(self.documents):
            #content_words = re.findall(r'\w+', content.lower())
            content_words=[self.stem_custom(w) for w in re.findall(r'\w+', content.lower())]
            content_word_counts = Counter(content_words)

            score = sum(
                (content_word_counts[word]/len(content_words))*self.idf.get(word, 1.0) 
                for word in keywords 
                if word in content_word_counts
            )
            scores.append((score, i))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for (score, idx) in scores[:3]]

        print("Selected documents (keyword-based):", top_indices)

        return "\n\n".join(self.documents[idx][1] for idx in top_indices)




    def _extract_keywords(self, text: str) -> list[str]:
        """Extracts meaningful keywords (removes stopwords)."""
        words = re.findall(r'\w+', text.lower())
        
        # Basic stopwords list (expand as needed)
        stop_set = {
            "what", "how", "why", "when", "where", "who", 
            "is", "are", "the", "a", "an", "and", "or", "of", "in", "to"
        }
        
        return [word for word in words if word not in stop_set]
    
    
