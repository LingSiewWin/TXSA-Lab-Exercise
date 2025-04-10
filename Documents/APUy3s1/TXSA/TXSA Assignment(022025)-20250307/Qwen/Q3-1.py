import sys
import argparse
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.tag import RegexpTagger
import nltk

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def main():
    parser = argparse.ArgumentParser(description='POS Tagging Demonstration')
    parser.add_argument('input_file', nargs='?', default='Data_2.txt', help='Input text file (default: Data_2.txt)')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as file:
            sentence = file.read().strip()
            if not sentence:
                print("Error: Input file is empty")
                sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        sys.exit(1)

    tokens = word_tokenize(sentence)

    # NLTK POS tagging
    nltk_tags = nltk.pos_tag(tokens)
    print("NLTK POS Tags:")
    print(nltk_tags)

    # TextBlob POS tagging
    blob = TextBlob(sentence)
    textblob_tags = blob.tags
    print("\nTextBlob POS Tags:")
    print(textblob_tags)

    # Regex Tagger
    patterns = [
        (r'^[Tt]he$', 'DT'),
        (r'^[Bb]ig|[Bb]lack|[Ww]hite$', 'JJ'),
        (r'^dog|cat$', 'NN'),
        (r'^barked|chased$', 'VBD'),
        (r'^at$', 'IN'),
        (r'^and$', 'CC'),
        (r'^away$', 'RB'),
        (r'^\.$', '.'),
    ]
    regex_tagger = RegexpTagger(patterns)
    regex_tags = regex_tagger.tag(tokens)
    print("\nRegex Tagger POS Tags:")
    print(regex_tags)

if __name__ == "__main__":
    main()