import numpy as np
import gensim
import json

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        for i, line in enumerate(f):
            splited = line.split("*")
            tokens = gensim.utils.simple_preprocess(splited[1])
            card_dict[splited[0]] = tokens
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


card_dict = {}
card_corpus = read_corpus("CardTexts.txt")
model = gensim.models.doc2vec.Doc2Vec(vector_size=10, min_count=1, epochs=50)
model.build_vocab(card_corpus)
model.train(card_corpus, total_examples=model.corpus_count, epochs=model.epochs)

result = {}
for key, val in card_dict.items():
    embedding_np = model.infer_vector(val).astype(np.float64)
    result[key] = list(embedding_np)

with open("CardEmbeddings.json", 'w') as f:
    json.dump(result, f)
