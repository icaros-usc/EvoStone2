import numpy as np
import gensim
import json
import pandas as pd

class Card2Vec():
    def __init__(self,corpus_file, embedding_size=15, epochs=20):
        self.embeddings = []
        self.corpus_file = corpus_file
        self.embedding_size = embedding_size
        self.card_dict = {}
        self.epochs = epochs

    def read_corpus(self, tokens_only=False):
        self.card_texts = pd.read_csv(self.corpus_file)
        for i, row in self.card_texts.iterrows():
            tokens = gensim.utils.simple_preprocess(row["description"])
            self.card_dict[row["card_name"]] = tokens
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def get_card_embeddings(self):
        card_corpus = self.read_corpus()
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=self.embedding_size, min_count=1, epochs=self.epochs)
        self.model.build_vocab(card_corpus)
        self.model.train(card_corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        for index, row in self.card_texts.iterrows():
            result = {}
            embedding_np = self.model.infer_vector(self.card_dict[row["card_name"]]).astype(np.float64)
            result["cardName"] = row["card_name"]
            result["embedding"] = list(embedding_np)
            self.embeddings.append(result)

        with open("CardEmbeddings.json", 'w') as f:
            json.dump(self.embeddings, f)

# model = Card2Vec("CardTexts.csv", 15)
# model.get_card_embeddings()
