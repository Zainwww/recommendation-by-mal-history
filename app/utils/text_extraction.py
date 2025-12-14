import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import yake
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words_indonesian = stopwords.words('indonesian')
stop_words_english = stopwords.words('english')

def simple_sentence_tokenize(text):
    return re.split(r'[.!?]\s+', text)

def extract_tfidf_keywords(text, top_k=10):
    vectorizer = TfidfVectorizer(stop_words=stop_words_english)
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]

    idx = np.argsort(scores)[::-1][:top_k]
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = feature_names[idx]

    return list(keywords)


model = SentenceTransformer('distilbert-base-nli-mean-tokens')
def extract_bert_keywords(text, top_k=10):
    words = list(set(text.lower().split()))
    words = [w for w in words if len(w) > 3]

    text_emb = model.encode([text])[0]
    word_embs = model.encode(words)

    sims = util.cos_sim(text_emb, word_embs)[0].cpu().numpy()
    idx = np.argsort(sims)[::-1][:top_k]

    return [words[i] for i in idx]

def extract_bert_keywords_batch(texts, top_k=10, batch_size=32):
    results = []
    texts = [t if isinstance(t, str) and t.strip() else "" for t in texts] # pastikan semua string

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # encode full text
        text_embeddings = model.encode(batch_texts)

        for text, text_emb in zip(batch_texts, text_embeddings):
            if not text:
                results.append([])
                continue

            words = list(set(text.lower().split()))
            words = [w for w in words if len(w) > 3]

            if not words:
                results.append([])
                continue

            word_embs = model.encode(words)
            similarities = cosine_similarity([text_emb], word_embs)[0]
            top_idx = similarities.argsort()[-top_k:][::-1]

            results.append([words[i] for i in top_idx])

    return results

def extract_yake_keywords(text, top_k=10):
    kw_extractor = yake.KeywordExtractor(lan="id", n=1, top=top_k)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

def extract_rake_keywords(text, top_k=10):
    rake = Rake(
        stopwords=stop_words_english,
        sentence_tokenizer=simple_sentence_tokenize,
    )
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords[:top_k]