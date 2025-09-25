"""
This code is utils function for speech GR task.

Currently include code for constrained beam search.
"""
from typing import List


class Trie:
    def __init__(self):
        self.children = {}

class RestrictDecodeVocab:
    def __init__(self, valid_ids: List[str], tokenizer):
        self.valid_ids = valid_ids
        self.tokenizer = tokenizer
        self.tokenized_valid_ids = [
            tokenizer.encode(docid)
            for docid in self.valid_ids
        ]

        self.trie = Trie()
        
        for token_seq in self.tokenized_valid_ids:
            node = self.trie
            for token in token_seq:
                if token not in node.children:
                    node.children[token] = Trie()
                node = node.children[token]

    def __call__(self, batch_idx, prefix_beam):
        valid_tokens = []
        
        prefix_beam = prefix_beam.tolist()
        
        if len(prefix_beam) == 1 and prefix_beam[0] == 0:
            # return list(self.trie.children.keys())
            valid_tokens = list(self.trie.children.keys())
        elif len(prefix_beam) == 1 and prefix_beam[0] != 0:
            raise Exception("Unexpected behavior")
        else:
            node = self.trie
            
            for beam in prefix_beam[1:]:
                node = node.children.get(beam)
                if node is None:
                    break
                
            if node is not None:
                valid_tokens = list(node.children.keys())
            else:
                valid_tokens = [self.tokenizer.eos_token_id]
                
        return valid_tokens