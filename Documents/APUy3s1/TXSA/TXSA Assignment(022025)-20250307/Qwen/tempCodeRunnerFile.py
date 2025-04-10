import sys
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk import ChartParser

nltk.download('punkt', quiet=True)

def main():
    parser = argparse.ArgumentParser(description='Parse Tree Generation')
    parser.add_argument('input_file', nargs='?', default='Q3/Data_2.txt', help='Input text file (default: Data_2.txt)')
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

    tokens = word_tokenize(sentence.replace('.', ''))  # Remove period for parsing

    grammar = nltk.CFG.fromstring("""
        S -> NP VP
        NP -> DT AdjPhrase NN
        AdjPhrase -> JJ | JJ AdjPhrase
        VP -> VBD PP ConjVP | VBD ADVP
        ConjVP -> CONJ VP
        PP -> IN NP
        ADVP -> RB
        CONJ -> 'and'
        DT -> 'The' | 'the'
        JJ -> 'big' | 'black' | 'white'
        NN -> 'dog' | 'cat'
        VBD -> 'barked' | 'chased'
        IN -> 'at'
        RB -> 'away'
    """)

    parser = ChartParser(grammar)
    parse_trees = list(parser.parse(tokens))

    print(f"\nOriginal Sentence:\n\"{sentence}\"\n")
    print("Parse Trees:")
    if parse_trees:
        for tree in parse_trees:
            tree.pretty_print()
    else:
        print("No valid parse trees found. Ensure the sentence matches the CFG grammar.")

if __name__ == "__main__":
    main()