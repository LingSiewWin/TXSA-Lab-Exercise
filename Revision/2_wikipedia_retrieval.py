import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get user query
query = input("Enter a search query: ")

# Fetch related Wikipedia summaries (top 5)
try:
    search_results = wikipedia.search(query, results=5)
except:
    print("Error fetching from Wikipedia")
    exit()

docs = []
titles = []

for title in search_results:
    try:
        summary = wikipedia.summary(title, sentences=3)
        docs.append(summary)
        titles.append(title)
    except:
        continue

# Add the query as the first item to compare
docs.insert(0, query)
titles.insert(0, "QUERY")

# Vectorize all texts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(docs)

# Compute similarity between query and docs
cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

# Sort by relevance
ranked_docs = sorted(zip(titles[1:], cosine_scores), key=lambda x: x[1], reverse=True)

print("\nTop Matching Results:")
for title, score in ranked_docs:
    print(f"{title} (score: {score:.4f})")