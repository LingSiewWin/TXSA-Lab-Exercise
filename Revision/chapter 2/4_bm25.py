import wikipedia
from rank_bm25 import BM25Okapi
import re

def preprocess(text):
    # Remove special characters and lowercase
    return re.sub(r'\W+', ' ', text).lower().split()

# Get user input
query = input("Enter a search query: ")

# Fetch related Wikipedia summaries (top 5)
try:
    search_results = wikipedia.search(query, results=5)
except Exception as e:
    print("Error fetching from Wikipedia:", e)
    exit()

docs = []
titles = []

for title in search_results:
    try:
        summary = wikipedia.summary(title, sentences=3)
        tokens = preprocess(summary)
        docs.append(tokens)
        titles.append(title)
    except Exception as e:
        continue

if not docs:
    print("No documents found.")
    exit()

# Tokenize the query
tokenized_query = preprocess(query)

# Build BM25 model
bm25 = BM25Okapi(docs)

# Get BM25 scores for each document
scores = bm25.get_scores(tokenized_query)

# Rank and display top results
ranked_docs = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)

print("\nTop Matching Results (BM25 Score):")
for title, score in ranked_docs:
    print(f"{title} (score: {score:.4f})")