import string
import csv
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.vocab_size = 0
        self.merge_history = []
    
    def clean_text(self, text, lowercase=True, strip_punct=False):
        if lowercase:
            text = text.lower()
        if strip_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def tokenize(self, text):
        """Tokenize text using learned BPE merges"""
        text = self.clean_text(text)
        words = text.split()
        tokens = []
        
        for word in words:
            current = list(word + '</w>')
            
            changed = True
            while changed and len(current) > 1:
                changed = False
                i = 0
                while i < len(current) - 1:
                    pair = (current[i], current[i + 1])
                    if pair in self.merges:
                        current = current[:i] + [self.merges[pair]] + current[i + 2:]
                        changed = True
                    else:
                        i += 1
            
            tokens.extend(current)
        
        return tokens


    def train(self, text, target_vocab_size, show_progress=True, max_steps=20):
        text = self.clean_text(text)
        words = text.split()
        
        if show_progress:
            print(f"Training on {len(words)} words, {len(set(words))} unique")
            print(f"Target vocab: {target_vocab_size}")
            print("-" * 50)
        
        word_counts = Counter(words)
        word_counts = {word + '</w>': count for word, count in word_counts.items()}
        
        chars = set()
        for word in word_counts.keys():
            chars.update(word)
        self.vocab = {char: idx for idx, char in enumerate(sorted(chars))}
        self.vocab_size = len(self.vocab)
        
        splits = {word: list(word) for word in word_counts.keys()}
        self.merges = {}
        self.merge_history = []
        
        step = 0
        while self.vocab_size < target_vocab_size and step < 200:
            pair_counts = defaultdict(int)
            for word, split in splits.items():
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_counts[pair] += word_counts[word]
            
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            count = pair_counts[best_pair]
            
            if count < 2 and step > 15:
                break
                
            new_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab_size
            self.vocab_size += 1
            
            self.merge_history.append({
                'step': step + 1,
                'pair': best_pair,
                'new_token': new_token,
                'count': count
            })
            
            splits = self._apply_merge(best_pair[0], best_pair[1], new_token, splits, word_counts)
            
            if show_progress and step < max_steps:
                if step % 4 == 0:
                    print(f"Step {step + 1}: merge {best_pair} → '{new_token}' ({count}x)")
                    self._show_samples(splits, word_counts, 2)
                    print()
                else:
                    print(f"Step {step + 1}: merge {best_pair} → '{new_token}' ({count}x)")
            
            step += 1
        
        if show_progress:
            print(f"\nFinal: {self.vocab_size} tokens, {len(self.merges)} merges")
            print("\nTop 20 merges:")
            for i, merge in enumerate(self.merge_history[:20]):
                print(f"  {i+1:2d}. {merge['pair']} → '{merge['new_token']}'")
            
            print(f"\nVocabulary size: {self.vocab_size}")
            print(f"Total merges: {len(self.merges)}")
        
        return splits
    
    def _apply_merge(self, a, b, new_token, splits, word_counts):
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits
    
    def _show_samples(self, splits, word_counts, sample_count=3):
        shown = 0
        for word, split in splits.items():
            if shown >= sample_count:
                break
            clean_word = word.replace('</w>', '')
            print(f"  {clean_word}: {split}")
            shown += 1
    
    def show_vocab(self, max_items=50):
        print(f"\nVocabulary (showing first {max_items} items):")
        print("-" * 40)
        items_shown = 0
        for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
            if items_shown >= max_items:
                break
            print(f"  {idx:3d}: '{token}'")
            items_shown += 1

def read_corpus_from_csv(filename, text_column='text'):
    """Read corpus from CSV file"""
    corpus_text = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if text_column in row:
                    corpus_text.append(row[text_column])
        return ' '.join(corpus_text)
    except FileNotFoundError:
        print(f"Error: CSV file '{filename}' not found")
        return ""
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return ""

def run_training():
    tokenizer = BPETokenizer()
    
    corpus = read_corpus_from_csv('corpus.csv')
    
    if not corpus:
        print("No corpus data found. Using fallback text.")
        corpus = """
        machine learning deep neural networks require extensive training data
        artificial intelligence transforms industries through automation
        natural language processing enables computers to understand human language
        programming languages like python javascript java are widely used
        software development involves designing coding testing applications
        """
    
    print("Training BPE tokenizer")
    print("=" * 50)
    
    splits = tokenizer.train(corpus, 2000, max_steps=15)
    
    tokenizer.show_vocab(60)
    
    test_words = ["artificial", "intelligence", "programming", "development", "algorithm", "learning"]
    print("\nTokenization examples:")
    print("-" * 30)
    for word in test_words:
        tokens = tokenizer.tokenize(word)
        print(f"{word:15} → {tokens}")

run_training()
