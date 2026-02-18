import torch
from transformers import LogitsProcessor
from typing import List, Dict, Any, Iterable

class Trie:
    def __init__(self, sequences: Iterable[Iterable[int]] = None):
        self.trie = {}
        if sequences:
            for seq in sequences:
                self.insert(seq) 

    def insert(self, sequence: Iterable[int]):
        node = self.trie
        for token in sequence:
            token = int(token)
            if token not in node:
                node[token] = {}
            node = node[token]
        node[None] = True # End of sequence marker

    def get_children(self, prefix: Iterable[int]) -> List[int]:
        node = self.trie
        for token in prefix:
            token = int(token)
            if token not in node:
                return []
            node = node[token]
        
        return [k for k in node.keys() if k is not None]

class TrieLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that restricts generation to a prefix tree (Trie).
    Used to ensure the model only generates valid document IDs.
    """
    def __init__(self, trie: Trie, start_token_id: int):
        self.trie = trie
        self.start_token_id = start_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: The generated sequence so far (B, L)
            scores: The logits for the next token (B, V)
        """
        # input_ids includes the start_token_id
        # We need to extract the sequence AFTER the start_token_id for the trie
        
        mask = torch.full_like(scores, float("-inf"))
        
        for i in range(input_ids.shape[0]):
            # Find the position of the last start_token_id (if there are multiple)
            # or just assume the sequence starts at index 0 or 1.
            # Usually for Seq2Seq, index 0 is BOS.
            
            # Simple assumption: sequence starts at index 1
            curr_seq = input_ids[i, 1:].tolist()
            
            valid_next = self.trie.get_children(curr_seq)
            
            if valid_next:
                mask[i, valid_next] = 0
            else:
                # If no valid next (should not happen if Trie is built correctly 
                # and max_length is respected), allow EOS or something.
                pass
                
        return scores + mask
