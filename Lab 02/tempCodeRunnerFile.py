import re
import nltk
import matplotlib.pyplot as plt
import urllib.request
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

# Download necessary corpora
nltk.download('punkt')
#new
import shutil
import nltk
import os

nltk_data_path = os.path.expanduser("~/nltk_data")
if os.path.exists(nltk_data_path):
    shutil.rmtree(nltk_data_path)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('treebank')
nltk.download('udhr')
nltk.download('toolbox')

nltk.download('punkt_tab') 
nltk.download('average_perceptron_tagger')
nltk.download('universal_tagset')

nltk.download('treebank')
nltk.download('udhr')
nltk.download('toolbox')

# Function to count vowels in a word
def count_vowels(word):
    vowels = re.findall(r'[aeiou]', word, re.IGNORECASE)
    return len(vowels), vowels

# Function to visualize vowel frequency
def plot_vowel_distribution(text):
    vowels = re.findall(r'[aeiou]', text, re.IGNORECASE)
    fdist = FreqDist(vowels)
    fdist.plot(title="Vowel Frequency Distribution")

# Function to perform stemming using regex
def regex_stem(word):
    pattern = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(pattern, word)[0]
    return stem

# Function to analyze a text corpus
def analyze_corpus(text):
    tokens = word_tokenize(text.lower())
    stems = [regex_stem(token) for token in tokens]
    return stems

# Function to extract vowel clusters from a dataset
def extract_vowel_clusters():
    wsj = sorted(set(nltk.corpus.treebank.words()))
    fd = FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word))
    return fd

# Function to scrape data from Project Gutenberg
def scrape_gutenberg(url, char_limit=300):
    with urllib.request.urlopen(url) as f:
        return f.read(char_limit).decode('utf-8')

# Function to analyze Rotokas language patterns
def analyze_rotokas():
    rotokas_words = nltk.corpus.toolbox.words('rotokas.dic')
    cvs = [cv for w in rotokas_words for cv in re.findall(r'[ptksvr][aeiou]', w)]
    cfd = nltk.ConditionalFreqDist(cvs)
    cfd.tabulate()

# Sample text for processing
sample_text = "Supercalifragilisticexpialidocious is a fantastic word!"
stems = analyze_corpus(sample_text)
print("Stems:", stems)

# Plot vowel distribution
plot_vowel_distribution(sample_text)

# Scrape data from Project Gutenberg
gutenberg_data = scrape_gutenberg("http://www.gutenberg.org/ebooks/2554?msg=welcome_stranger")
print("Gutenberg Sample:", gutenberg_data)

# Analyze Rotokas language
analyze_rotokas()

# Extract and display vowel clusters
vowel_clusters = extract_vowel_clusters()
print("Most Common Vowel Clusters:", vowel_clusters.most_common(10))
