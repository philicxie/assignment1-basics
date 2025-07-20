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

def bytes2pair(ia, ib):
    return bytes([ia]), bytes([ib])


class BPETokenizer:
    def __init__(self, input_path, vocab_size, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]

        self.merges = []
        self.input_path = input_path

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {i: bytes([i]) for i in range(DEFAULT_TOKEN_NUM)}
        for i, token in enumerate(special_tokens):
            self.vocab[DEFAULT_TOKEN_NUM + i] = token.encode("utf-8")

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