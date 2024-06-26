
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from collections import defaultdict
import networkx as nx


import nltk
nltk.download('punkt')


text = "The heart size and pulmonary vascularity appear within normal limits." \
       " A large hiatal hernia is noted. The lungs are free of focal airspace disease." \
       " No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.";
#(Yake) library selects the most important keywords using the text statistical features method from the article.

def getkewords_yake(text):
    import yake
    kw_extractor = yake.KeywordExtractor()

    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 5
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    # for kw in keywords:
    #     print(kw)
    #     print(kw[0])


#perform keyword extraction from medical text in Python using the Natural Language Toolkit (NLTK) and the TextRank algorithm:
def get_medical_list(file_path):
        # load medical words

        with open(file_path, 'r', encoding="utf-8") as f:
            words = f.readlines()
            med_set = set(m.strip() for m in words)
            return list(frozenset(med_set))

# nltk.download('punkt')
def getkewords(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]

    # Filter out stop words and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    filtered_words = [word for word in words if word not in stop_words]

    # Calculate word frequency
    word_freq = defaultdict(int)
    for word in filtered_words:
        word_freq[word] += 1

    # Calculate weighted word frequency
    max_freq = max(word_freq.values())
    weighted_word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    # Build a graph of words using TextRank algorithm
    graph = nx.Graph()
    for i, word in enumerate(filtered_words):
        for j in range(i + 1, len(filtered_words)):
            if word != filtered_words[j]:
                graph.add_edge(word, filtered_words[j])

    pagerank_scores = nx.pagerank(graph, weight='weight')

    # Sort the words by their TextRank scores and extract the top 10 keywords
    keywords = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:10]

    print(keywords)
    # improve the result
    # Extract medical keywords from the document
    # use Named Entity Recognition (NER) using spaCy to extract medical keywords from text in Python:
    import spacy

    # Load the pre-trained model for English
    nlp = spacy.load("en_core_web_sm")
    # Process the text with the NLP model
    doc = nlp(text)
    medical_keywords = []

    # file contain common chest  medical words

    medical_words = get_medical_list("medical.txt")

    for token in doc:
        nn = token.ent_type_
        if token.text in medical_words:
            medical_keywords.append(token.text)

    # print(medical_keywords)
    # print(keywords)

    final_keywords = keywords + medical_keywords
    # print(final_keywords)

    final_keywords = list(set(final_keywords))
    # print("finalll  resultt")

    # print(final_keywords)

    return final_keywords

#getkewords_yake(text)
# keywords=getkewords(text)

