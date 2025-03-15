import os
import re
import pandas as pd
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()

def load_text_files(directory):
    """Load all .txt files from a directory."""
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith("Data_1.txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                texts.append((filename, f.read()))
    return texts

def regex_stem(word):
    """Simple regex-based stemmer (removes common suffixes)."""
    return re.sub(r'(ing|ed|es|s|able|ly)$', '', word, flags=re.IGNORECASE)

def process_text(text):
    """Tokenize and lowercase text."""
    return [word.lower() for word in word_tokenize(text) if word.isalpha()]

def apply_stemmers(tokens):
    """Apply all stemmers and return results."""
    return {
        "Original": tokens,
        "Regex Stemmer": [regex_stem(word) for word in tokens],
        "Porter Stemmer": [porter.stem(word) for word in tokens],
        "Lancaster Stemmer": [lancaster.stem(word) for word in tokens],
    }

def save_results(results, output_dir="stemmed_output"):
    """Save results to CSV files for comparison."""
    os.makedirs(output_dir, exist_ok=True)
    for filename, data in results.items():
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, f"stemmed_{filename}.csv"), index=False)

def main():
    # Load text files from the current directory
    texts = load_text_files(".")
    
    # Process each file and apply stemmers
    results = {}
    for filename, text in texts:
        tokens = process_text(text)
        stemmed = apply_stemmers(tokens)
        results[filename] = stemmed
    
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def apply_stemmers(tokens):
    return {
        "Original": tokens,
        "Regex Stemmer": [regex_stem(word) for word in tokens],
        "Porter Stemmer": [porter.stem(word) for word in tokens],
        "Lancaster Stemmer": [lancaster.stem(word) for word in tokens],
        "Lemmatizer": [lemmatizer.lemmatize(word) for word in tokens],  # Add this line
    }
    # Save results to CSV for comparison
    save_results(results)
    print(f"Stemming completed! Check the 'stemmed_output' directory for results.")

if __name__ == "__main__":
    main()
    
