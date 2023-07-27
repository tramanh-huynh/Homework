import gensim.downloader as api
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math

model = api.load("glove-twitter-200")
word = "beautiful"
print(model[word])

print("1----------")
result = model.most_similar(word, topn=10)
print(result)

#with numpy
def cosine_similarity_np(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def find_top_similar_words_np(model, word, topn=10):
    word_vector = model[word]
    similarity = [(w, cosine_similarity_np(word_vector, model[w])) for w in model.key_to_index]    
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn]

word = "beautiful"
similar_words_np = find_top_similar_words_np(model, word)
print(similar_words_np)

#without numpy
def cosine_similarity(vector1, vector2):
    dot = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
    magnitude1 = math.sqrt(sum([vector1[i] ** 2 for i in range(len(vector1))]))
    magnitude2 = math.sqrt(sum([vector2[i] ** 2 for i in range(len(vector2))]))
    return dot/(magnitude1 * magnitude2)

def find_top_similar_words(model, word, topn=10):
    word_vector = model[word]
    similarity = [(w, cosine_similarity(word_vector, model[w])) for w in model.key_to_index]    
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn]

word = "beautiful"
similar_words = find_top_similar_words(model, word)
print(similar_words)

print("2----------")
result = model.most_similar(positive=["january", "february"], topn=10)
print(result)

#with numpy
def find_top_similar_words_np2(model, words, topn=12):
    word_vectors = [model[word] for word in words]
    mean_vector = np.mean(word_vectors, axis=0)
    similarity = [(w, cosine_similarity_np(mean_vector, model[w])) for w in model.key_to_index]
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn]

words = ["january","february"]
similar_words_np2 = find_top_similar_words_np2(model, words, topn=12)
print(similar_words_np2)

#without numpy
def find_top_similar_words2(model, words, topn=12):
    word_vector = [model[word] for word in words]
    mean_vector = sum(word_vector) / len(word_vector)
    similarity = [(w, cosine_similarity(mean_vector, model[w])) for w in model.key_to_index]    
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn]

words = ["january","february"]
similar_words2 = find_top_similar_words2(model, words, topn=12)
print(similar_words2)


print("3----------")
result = model.similarity("man", "woman")
print(result)

#with numpy
sim_np = cosine_similarity_np(model["man"],model["woman"])
print(sim_np)
#without numpy
sim = cosine_similarity(model["man"],model["woman"])
print(sim)

print("4----------")
result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=1)
print(result)

#with numpy
def find_top_similar_words_np4(model, pos_words,neg_words, topn=1):
    pos_vectors = [model[word] for word in pos_words]
    neg_vectors = [model[word] for word in neg_words]
    mean_vector = np.mean(pos_vectors, axis=0) - np.mean(neg_vectors, axis=0)
    similarity = [(w, cosine_similarity_np(mean_vector, model[w])) for w in model.key_to_index]    
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn] 
    
pos_words = ["woman","king"]
neg_words = ["man"]
similar_words_np4 = find_top_similar_words_np4(model, pos_words, neg_words, topn=1)
print(similar_words_np4)


#without numpy
def find_top_similar_words4(model, pos_words,neg_words, topn=1):
    pos_vectors = [model[word] for word in pos_words]
    neg_vectors = [model[word] for word in neg_words]
    mean_vector = sum(pos_vectors) / len(pos_vectors) - sum(neg_vectors) / len(neg_vectors)
    similarity = [(w, cosine_similarity(mean_vector, model[w])) for w in model.key_to_index]    
    similarity.sort(key=lambda x: x[1], reverse=True)
    return similarity[:topn]    

pos_words = ["woman","king"]
neg_words = ["man"]
similar_words4 = find_top_similar_words4(model, pos_words, neg_words, topn=1)
print(similar_words4)

print("5----------")
result = model.most_similar(positive=["berlin", "vietnam"], negative=["hanoi"], topn=1)
print(result)

#with numpy
similar_words_np5 = find_top_similar_words_np4(model, ["berlin", "vietnam"], ["hanoi"], topn=1)
print(similar_words_np5)

#without numpy
similar_words5 = find_top_similar_words4(model, ["berlin", "vietnam"], ["hanoi"], topn=1)
print(similar_words5)

print("6----------")
result = model.similarity("marriage", "happiness")
print(result)

#with numpy
sim2_np = cosine_similarity_np(model["marriage"],model["happiness"])
print(sim2_np)
#without numpy
sim2 = cosine_similarity(model["marriage"],model["happiness"])
print(sim2)


