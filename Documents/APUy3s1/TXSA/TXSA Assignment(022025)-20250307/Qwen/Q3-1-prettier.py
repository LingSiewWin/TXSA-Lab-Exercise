import sys
import argparse
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.tag import RegexpTagger
import nltk
from tabulate import tabulate
from colorama import Fore, Style

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

def main():
    parser = argparse.ArgumentParser(description='POS Tagging Demonstration (4.0 Edition)')
    parser.add_argument('input_file', nargs='?', default='Data_2.txt', 
                       help='Input text file (default: Data_2.txt). Format: One sentence per line.')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as file:
            sentence = file.read().strip()
            if not sentence:
                raise ValueError("Input file is empty")
    except FileNotFoundError:
        sys.exit(f"{Fore.RED}Error: File '{args.input_file}' not found{Style.RESET_ALL}")
    except Exception as e:
        sys.exit(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

    tokens = word_tokenize(sentence)

    # NLTK POS tagging
    nltk_tags = nltk.pos_tag(tokens)
    
    # TextBlob: Align with NLTK tokenization [[4]][[8]]
    reconstructed_sentence = ' '.join(tokens)
    blob = TextBlob(reconstructed_sentence)
    textblob_tags = blob.tags
    
    # Validate tag list lengths [[10]]
    if len(nltk_tags) != len(textblob_tags):
        print(f"{Fore.YELLOW}Warning: Tokenization mismatch between NLTK and TextBlob! "
              f"Truncating to {min(len(nltk_tags), len(textblob_tags))} tokens{Style.RESET_ALL}")
        min_len = min(len(nltk_tags), len(textblob_tags))
        nltk_tags = nltk_tags[:min_len]
        textblob_tags = textblob_tags[:min_len]
        tokens = tokens[:min_len]

    # Regex Tagger
    regex_tagger = RegexpTagger([
        (r'^[Tt]he$', 'DT'),
        (r'^[Bb]ig|[Bb]lack|[Ww]hite$', 'JJ'),
        (r'^dog|cat$', 'NN'),
        (r'^barked|chased$', 'VBD'),
        (r'^at$', 'IN'),
        (r'^and$', 'CC'),
        (r'^away$', 'RB'),
        (r'^\.$', '.'),
    ])
    regex_tags = regex_tagger.tag(tokens)

    # Generate comparison table
    table = []
    for i, word in enumerate(tokens):
        row = [word]
        # Add NLTK tag with color coding
        row.append(f"{Fore.GREEN}{nltk_tags[i][1]}{Style.RESET_ALL}" 
                   if nltk_tags[i][1] == textblob_tags[i][1] 
                   else f"{Fore.YELLOW}{nltk_tags[i][1]}{Style.RESET_ALL}")
        # Add TextBlob tag
        row.append(textblob_tags[i][1])
        # Add Regex tag with error highlighting
        regex_tag = regex_tags[i][1] if regex_tags[i][1] else f"{Fore.RED}None{Style.RESET_ALL}"
        row.append(regex_tag)
        table.append(row)

    # Print results with enhanced formatting
    print(f"\n{Fore.CYAN}Original Sentence:{Style.RESET_ALL}\n\"{sentence}\"\n")
    
    print(f"{Fore.CYAN}{'='*60}")
    print("Tagging Comparison Table")
    print(f"{'='*60}{Style.RESET_ALL}")
    print(tabulate(table, 
                  headers=["Word", "NLTK Tag", "TextBlob Tag", "Regex Tagger"], 
                  tablefmt="psql"))
    
    # Quantitative analysis
    regex_coverage = sum(1 for tag in regex_tags if tag[1]) / len(regex_tags) * 100
    agreement = sum(1 for a, b in zip(nltk_tags, textblob_tags) if a[1] == b[1]) / len(nltk_tags) * 100
    print(f"\n{Fore.CYAN}Analysis:{Style.RESET_ALL}")
    print(f"- Regex Tagger Coverage: {regex_coverage:.1f}%")
    print(f"- NLTK vs TextBlob Agreement: {agreement:.1f}%")

    # Explanation of differences
    print(f"\n{Fore.CYAN}Key Differences Explained:{Style.RESET_ALL}")
    print(f"1. {Fore.GREEN}NLTK{Style.RESET_ALL}: Statistical model with full Penn Treebank tags [[3]][[9]]")
    print(f"2. {Fore.BLUE}TextBlob{Style.RESET_ALL}: Simplified universal tags (e.g., 'NNP' → 'NN') [[5]][[6]]")
    print(f"3. {Fore.RED}Regex Tagger{Style.RESET_ALL}: Rule-based; fails on unknown patterns [[1]][[8]]")

if __name__ == "__main__":
    main()