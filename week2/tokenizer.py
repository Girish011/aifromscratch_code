"""BPE tokenizer shared by week2 notebooks.

Extracted from 8_tokenizer_from_scratch.ipynb so SFT / other notebooks can:
    from tokenizer import BPETokenizer
"""

import re
import json
import collections
from pathlib import Path

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size           # target size (chars + merges)
        self.merges = {}    # (token_a, token_b) → merged_token string
        self.vocab = {}     # token string → integer id
        self.inv_vocab = {} # integer id → token string

    def _word_to_tokens(self, word):
        # "cat" → ['c', 'a', 't', '</w>']
        return list(word) + ['</w>']

    def _init_vocab_from_text(self, text):
        """Count words; seed vocab with every character (+ </w>)."""
        # words + leftover punctuation as separate tokens
        words = re.findall(r'\b\w+\b|\S', text.lower())
        word_freq = collections.Counter(words)

        char_set = set()
        for word, freq in word_freq.items():
            for ch in word:
                char_set.add(ch)
        char_set.add('</w>')

        # sorted → stable ids across runs
        self.vocab = {ch: i for i, ch in enumerate(sorted(char_set))}
        self.inv_vocab = {i: ch for ch, i in self.vocab.items()}
        return word_freq

    def train(self, text):
        """Learn BPE merges from corpus until vocab reaches vocab_size."""
        word_freq = self._init_vocab_from_text(text)

        # Spaced spelling of each unique word's current tokens
        #   "newest" → "n e w e s t </w>"
        splits = {word: ' '.join(self._word_to_tokens(word)) for word in word_freq.keys()}

        # How many NEW merge tokens we still need
        for merge_step in range(self.vocab_size - len(self.vocab)):
            # Count adjacent pairs across the corpus (weighted by word freq)
            pair_counts = collections.defaultdict(int)
            for word, freq in word_freq.items():
                symbols = splits[word].split()
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            # New symbol name (toy rule: drop </w> from the right piece's spelling)
            # Keep </w> inside the merged piece so decode can restore spaces.
            # BUGFIX: old code stripped </w> → merges like ('s','</w>') became token "s"
            #         (collided with char "s") and decode glued words: "theapirate..."
            new_token = best_pair[0] + best_pair[1]

            # Register in vocab + remember this merge for encode()
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.inv_vocab[new_id] = new_token
            self.merges[best_pair] = new_token

            # Apply merge in every word's spaced spelling
            #   "... e s ..." → "... es ..." when best_pair == ('e','s')
            for word in splits:
                splits[word] = splits[word].replace(' '.join(best_pair), new_token)

            if merge_step % 100 == 0:
                print(f"Merge {merge_step}: {best_pair} -> {new_token} (vocab size {len(self.vocab)})")

        print(f"Training complete. Final vocab size: {len(self.vocab)}")

    def encode(self, text):
        """Text → list of token ids (apply learned merges in training order)."""
        words = re.findall(r'\b\w+\b|\S', text.lower())
        ids = []
        for word in words:
            tokens = self._word_to_tokens(word)  # start from characters + </w>

            # Greedily merge: always apply the EARLIEST-learned merge that matches
            while True:
                pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                merge_idx = float('inf')
                pair_to_merge = None
                for pair in pairs:
                    if pair in self.merges:
                        # earlier in self.merges.keys() = learned sooner = higher priority
                        idx = list(self.merges.keys()).index(pair)
                        if idx < merge_idx:
                            merge_idx = idx
                            pair_to_merge = pair
                if pair_to_merge is None:
                    break  # no more known merges apply

                new_token = self.merges[pair_to_merge]
                # Replace first occurrences of that pair left-to-right in this pass
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            # Map each final piece to its id
            for token in tokens:
                ids.append(self.vocab.get(token, self.vocab.get('<UNK>', 0)))
        return ids

    def decode(self, ids):
        """Token ids → string (</w> becomes a space between words)."""
        tokens = [self.inv_vocab.get(i, '<UNK>') for i in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def save(self, path):
        """Save vocab + merges so later notebooks can reload the SAME ids.

        Sticky: MiniGPT checkpoints are tied to this vocab size / id map.
        DPO must load this file — not retrain BPE on new text — or embed/head
        shapes will mismatch (e.g. 260 vs 277).
        """
        path = Path(path)
        payload = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            # merges keys are (a, b) tuples → store as lists for JSON
            "merges": [[a, b, merged] for (a, b), merged in self.merges.items()],
        }
        path.write_text(json.dumps(payload))
        print(f"Saved tokenizer → {path} (vocab={len(self.vocab)})")

    def load(self, path):
        """Restore vocab + merges from save(). Returns self for chaining."""
        path = Path(path)
        payload = json.loads(path.read_text())
        self.vocab_size = payload["vocab_size"]
        # token → id (JSON values are ints; normalize just in case)
        self.vocab = {t: int(i) for t, i in payload["vocab"].items()}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.merges = {(a, b): merged for a, b, merged in payload["merges"]}
        print(f"Loaded tokenizer ← {path} (vocab={len(self.vocab)})")
        return self
