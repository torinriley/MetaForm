import re
from collections import defaultdict, Counter

class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.bpe_vocab = {}
    
    class BPE:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.bpe_vocab = {}
            self.merges = []
    
    def fit(self, texts):
        """
        Train the BPE model on the provided texts.

        Args:
            texts (list of str): The list of texts to train on.
        """
        vocab = Counter()
        for text in texts:
            tokens = self._get_tokens(text)
            vocab.update(tokens)
        
        while len(self.merges) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
        
        self.bpe_vocab = {token: i for i, (token, _) in enumerate(vocab.items())}

    def tokenize(self, text):
        """
        Tokenize the input text using the learned BPE model.

        Args:
            text (str): The input text.

        Returns:
            list of str: The list of tokenized subword tokens.
        """
        tokens = self._get_tokens(text)
        output_tokens = []
        for token in tokens:
            subwords = self._bpe_encode(token)
            output_tokens.extend(subwords)
        return output_tokens

    def _get_tokens(self, text):
        # Basic whitespace tokenization
        return text.split()

    def _get_stats(self, vocab):
        """Get pair frequencies in the vocabulary."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """Merge the most frequent pair in the vocabulary."""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = new_vocab.get(new_word, 0) + vocab[word]
        return new_vocab

    def _bpe_encode(self, token):
        """Encode a token using BPE merges."""
        symbols = token.split()
        if len(symbols) < 2:
            return symbols
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            pair_found = False
            for pair in self.merges:
                if pair in pairs:
                    index = pairs.index(pair)
                    symbols[index] = ''.join(pair)
                    symbols.pop(index + 1)
                    pair_found = True
                    break
            if not pair_found:
                break
        return symbols
    
    def get_vocab(self):
        return self.bpe_vocab

def normalize_text(text):
    """
    Normalize the input text by lowercasing and removing non-alphanumeric characters.

    Args:
        text (str): The input text.

    Returns:
        str: The normalized text.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

class AdvancedTokenizer:
    def __init__(self, vocab_size=10000):
        self.bpe = BPE(vocab_size)
        self.vocab = {}
        self.reverse_vocab = {}
    
    def train(self, texts):
        """
        Train the tokenizer on a list of texts.

        Args:
            texts (list of str): The list of texts to train on.
        """
        normalized_texts = [normalize_text(text) for text in texts]
        
        self.bpe.fit(normalized_texts)
        
        self.vocab = self.bpe.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text):
        """
        Tokenize the input text.

        Args:
            text (str): The input text.

        Returns:
            list of int: The list of token IDs.
        """
        normalized_text = normalize_text(text)
        
        tokens = self.bpe.tokenize(normalized_text)
        
        return [self.vocab.get(token, self.vocab.get('[UNK]')) for token in tokens]
    
    def decode(self, token_ids):
        """
        Decode a list of token IDs back to text.

        Args:
            token_ids (list of int): The list of token IDs.

        Returns:
            str: The decoded text.
        """
        tokens = [self.reverse_vocab.get(id, '[UNK]') for id in token_ids]
        return ' '.join(tokens)
