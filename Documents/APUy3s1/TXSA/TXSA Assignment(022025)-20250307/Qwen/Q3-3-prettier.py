import sys
import argparse
import nltk
from nltk import CFG, ChartParser
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from colorama import Fore, Style

nltk.download('punkt', quiet=True)

class CFGParser:
    def __init__(self, grammar_str):
        self.grammar = CFG.fromstring(grammar_str)
        self.parser = ChartParser(self.grammar)
        
    def parse_sentence(self, tokens):
        return list(self.parser.parse(tokens))
    
    def analyze_ambiguity(self, trees):
        return len(trees) > 1
    
    def calculate_tree_depth(self, tree):
        return tree.height()
    
    def count_constituents(self, tree):
        return len(list(tree.subtrees()))

def main():
    parser = argparse.ArgumentParser(
        description='Advanced CFG Parse Tree Generator (4.0 Edition)'
    )
    parser.add_argument('input_file', nargs='?', default='Q3/Data_2.txt',
                       help='Input text file (default: Data_2.txt)')
    args = parser.parse_args()

    # Enhanced file handling with error recovery
    try:
        with open(args.input_file, 'r') as file:
            sentences = [line.strip() for line in file if line.strip()]
            if not sentences:
                raise ValueError("Input file contains no valid sentences")
    except FileNotFoundError:
        sys.exit(f"{Fore.RED}Error: File '{args.input_file}' not found{Style.RESET_ALL}")
    except Exception as e:
        sys.exit(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

    # Define multiple grammars for comparison [[5]][[7]]
    grammars = {
    'extended': """
        S -> NP VP | S Conj S
        NP -> DT AdjPhrase N | NP PP
        AdjPhrase -> Adj | Adj AdjPhrase
        VP -> V NP | VP PP | V ADVP | VP Conj VP  # Add VP coordination [[5]]
        PP -> P NP
        ADVP -> Adv
        Conj -> 'and'
        DT -> 'The' | 'the'
        Adj -> 'big' | 'black' | 'white'
        N -> 'dog' | 'cat'
        V -> 'barked' | 'chased'
        P -> 'at'
        Adv -> 'away'
    """
}

    # Process each sentence with analysis
    for i, sentence in enumerate(sentences):
        print(f"\n{Fore.CYAN}=== Analysis for Sentence {i+1} ==={Style.RESET_ALL}")
        print(f"Original: \"{sentence}\"")
        
        tokens = word_tokenize(sentence.lower())
        tokens = [t for t in tokens if t not in ('.', ',', ';')]  # Existing code  # Clean punctuation [[2]]
        
        # Compare multiple grammars
        for name, grammar in grammars.items():
            cfg_parser = CFGParser(grammar)
            trees = cfg_parser.parse_sentence(tokens)
            
            print(f"\n{Fore.YELLOW}Using {name.upper()} grammar:{Style.RESET_ALL}")
            if not trees:
                print(f"{Fore.RED}No valid parse trees found{Style.RESET_ALL}")
                continue
                
            # Structural analysis
            tree = trees[0]
            depth = cfg_parser.calculate_tree_depth(tree)
            constituents = cfg_parser.count_constituents(tree)
            ambiguous = cfg_parser.analyze_ambiguity(trees)
            
            # Visualization with ASCII art [[1]]
            print(f"\n{Fore.GREEN}Parse Tree (Depth: {depth}, Constituents: {constituents}):{Style.RESET_ALL}")
            tree.pretty_print()
            
            # Ambiguity analysis
            if ambiguous:
                print(f"{Fore.MAGENTA}Structural ambiguity detected!{Style.RESET_ALL}")
                print(f"Number of possible parses: {len(trees)}")
                
            # Linguistic explanation
            print(f"\n{Fore.BLUE}Linguistic Insights:{Style.RESET_ALL}")
            print(f"- Main constituents: {', '.join([str(n.label()) for n in tree])}")
            print(f"- Deepest branch: {max(tree.subtrees(), key=lambda t: t.height())}")

if __name__ == "__main__":
    main()