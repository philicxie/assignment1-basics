import collections

import regex as re

PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_TOKEN_NUM = 256


def initialize_pair_frequency(pretoken_freq: dict[str, int]) -> dict[tuple[bytes, bytes], int]:
    """
    takes in pretoken occurence table
    returns a pair frequency table
    we'll initialize the whole thing only once, and do modifications afterward
    """
    pair_freq = {}
    for pre_bytes, freq in pretoken_freq.items():
        for i in range(0, len(pre_bytes) - 1):
            pair = (pre_bytes[i], pre_bytes[i + 1])
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq
    return pair_freq

class BPETokenizer:
    def __init__(self, input_path=None, vocab_size=255, special_tokens=None):

        self.merges = []
        self.merge_dict = {}
        self.special_tokens = []
        self.input_path = input_path

        self.vocab_size = vocab_size
        if special_tokens:
            self.special_tokens = special_tokens
            for i, token in enumerate(special_tokens):
                self.vocab[DEFAULT_TOKEN_NUM + i] = token.encode("utf-8")
        self.encode_cache = {}
        self.vocab = {i: bytes([i]) for i in range(DEFAULT_TOKEN_NUM)}
        self.vocab_reverse = {}


    def init_from(self, vocab, merges, special_tokens=None):
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.merge_dict = {pair:idx for idx, pair in enumerate(merges)}
        self.vocab_reverse = {v:k for k, v in self.vocab.items()}

    def _process_chunk(self, chunk):
        res = []
        for m in re.finditer(PRETOKEN_PAT, chunk):
            if m.group(0) in self.encode_cache:
                return self.encode_cache[m.group(0)]
            # token_bytes = [c.encode("utf-8") for c in m.group(0).split()]
            token_bytes = [bytes([b]) for b in m.group(0).encode("utf-8")]
            res.extend(self._bpe_merge(token_bytes))

        return res

    def encode_iterable(self, text):
        for t in text:
            tokens = self.encode(t)
            yield from tokens


    def encode(self, text):
        ids = []
        if not self.special_tokens:
            ids.extend(self._process_chunk(text))
        else:
            split_pattern = re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")")
            idx = 0
            while idx < len(text):
                match = split_pattern.search(text, idx)
                if not match:
                    break
                chunk = text[idx:match.start()]
                ids.extend(self._process_chunk(chunk))
                ids.append(text[match.start():match.end()].encode("utf-8"))
                idx = match.end()
            if idx < len(text):
                chunk = text[idx:]
                ids.extend(self._process_chunk(chunk))
        print(ids)
        ans = [self.vocab_reverse[t] for t in ids if t in self.vocab_reverse]
        print("encode ans: ", ans)
        return ans

    def _bpe_merge(self, word):
        while True:
            if len(word) == 1:
                return word
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            bigram = min(pairs, key=lambda pair: self.merge_dict.get(pair, 2 ** 20))
            if bigram not in self.merge_dict:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word


    def decode(self, token_ids):
        tokens = [self.vocab[id] for id in token_ids]
        decoded_text = b''.join(tokens).decode("utf-8", errors='replace')
        return decoded_text



    def train(self):
        pair_freq = {}
        pretoken_freq = collections.Counter()
        with open(self.input_path, "r", encoding="utf-8") as f:
            content = f.read()
            split_pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            chunks = re.split(split_pattern, content)
            for chunk in chunks:
                if chunk in self.special_tokens:
                    continue
                tokens = re.findall(PRETOKEN_PAT, chunk)
                btokens = [tuple(bytes([bt]) for bt in list(t.encode("utf-8"))) for t in tokens]
                pretoken_freq.update(btokens)
            pair_freq = initialize_pair_frequency(pretoken_freq)
        print("Initializing BPE tokenizer...")

        while len(self.vocab) < self.vocab_size:
            most_freq = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
            self.merges.append(most_freq)
            new_vocab_id = max(self.vocab.keys()) + 1
            new_token = b"".join(most_freq)
            self.vocab[new_vocab_id] = new_token

            pretoken_count = 0
            pair_updates = 0
            new_pretoken_freq = {}

            for pretoken, freq in pretoken_freq.items():
                pretoken_count += 1
                new_pretoken = list(pretoken)
                i = 0
                modified = False
                while i < len(new_pretoken) - 1:
                    pair = (new_pretoken[i], new_pretoken[i + 1])

                    if pair == most_freq:
                        pair_updates += 1
                        if i > 0:
                            old_pair = (new_pretoken[i - 1], new_pretoken[i])
                            pair_freq[old_pair] = pair_freq.get(old_pair, 0) - freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]
                        if i < len(new_pretoken) - 2:
                            old_pair = (new_pretoken[i+1], new_pretoken[i+2])
                            pair_freq[old_pair] = pair_freq.get(old_pair, 0) - freq
                            if pair_freq[old_pair] <= 0:
                                del pair_freq[old_pair]

                        new_pretoken[i:i+2] = [new_token]
                        modified = True

                        if i > 0:
                            new_pair = (new_pretoken[i - 1], new_token)
                            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                        if i < len(new_pretoken) - 1:
                            new_pair = (new_token, new_pretoken[i+1])
                            pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq
                    i += 1
                if modified:
                    new_pretoken_freq[tuple(new_pretoken)] = freq
                else:
                    new_pretoken_freq[pretoken] = freq

            del pair_freq[most_freq]
            pretoken_freq = new_pretoken_freq

        return self.vocab, self.merges