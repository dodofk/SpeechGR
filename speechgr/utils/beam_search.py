import torch
from typing import List, Tuple, Optional

class BeamSearch:
    """
    Simple beam search implementation for UnitySpeechModel's retrieval phase.
    """
    def __init__(self, beam_width: int, max_len: int, pad_token_id: int, eos_token_id: int):
        self.beam_width = beam_width
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def search(
        self, 
        start_token_id: int, 
        model_fn, 
        memory: torch.Tensor, 
        device: torch.device,
        logits_processor=None
    ) -> torch.Tensor:
        """
        Args:
            start_token_id: ID to start generation (e.g., [START])
            model_fn: A function (curr_seq, memory) -> next_token_logits
            memory: Encoder/Decoder memory to condition on [B, MemLen, D]
            device: torch device
            logits_processor: Optional function to process logits (e.g., Trie constraint)

        Returns:
            Top-K sequences [B, beam_width, max_len]
        """
        batch_size = memory.size(0)
        
        # Initialize beams: (score, sequence)
        # Sequence shape: [Length]
        # We maintain a list of completed hypotheses and active beams for each batch item.
        
        # Simplified batch processing:
        # Structure: active_beams[b] = List of (score, tensor_seq)
        
        active_beams = [
            [(0.0, torch.tensor([start_token_id], device=device, dtype=torch.long))]
            for _ in range(batch_size)
        ]
        
        completed_beams = [[] for _ in range(batch_size)]
        
        for step in range(self.max_len):
            # Prepare inputs for the model
            # We need to flatten the batch to run efficiently: [B * n_beams, L]
            
            flat_input_seqs = []
            flat_memory_indices = [] # To select correct memory for each beam
            
            mapping = [] # (batch_idx, beam_idx_in_batch)
            
            has_active = False
            for b_idx in range(batch_size):
                for beam_idx, (score, seq) in enumerate(active_beams[b_idx]):
                    if seq[-1].item() == self.eos_token_id:
                        continue # Already done
                        
                    flat_input_seqs.append(seq)
                    flat_memory_indices.append(b_idx)
                    mapping.append((b_idx, beam_idx))
                    has_active = True
            
            if not has_active:
                break
                
            # Create batch tensors
            # Pad sequences to same length for this step (they should be same length anyway)
            batch_seqs = torch.stack(flat_input_seqs) # [N_active, L]
            batch_mem_indices = torch.tensor(flat_memory_indices, device=device)
            batch_memory = memory[batch_mem_indices] # [N_active, MemLen, D]
            
            # Forward Pass
            # Logits: [N_active, 1, V] (for the last step)
            logits = model_fn(batch_seqs, batch_memory) 
            logits = logits[:, -1, :] # [N_active, V]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Apply Constraints
            if logits_processor is not None:
                # Process expects (input_ids, scores)
                # input_ids: [N_active, L]
                log_probs = logits_processor(batch_seqs, log_probs)
                
            # Expansion
            # We need to distribute these results back to their respective batch items
            
            # Group by batch index
            candidates = [[] for _ in range(batch_size)]
            
            vocab_size = log_probs.size(-1)
            
            # For each active beam, get top-K next tokens
            # Optimization: only take top-K globally for the beam?
            # Standard: take top-K tokens for each hypothesis, then prune.
            
            topk_log_probs, topk_indices = torch.topk(log_probs, self.beam_width, dim=-1)
            
            for i, (b_idx, beam_idx) in enumerate(mapping):
                current_score = active_beams[b_idx][beam_idx][0]
                current_seq = active_beams[b_idx][beam_idx][1]
                
                for k in range(self.beam_width):
                    next_score = topk_log_probs[i, k].item()
                    next_token = topk_indices[i, k].item()
                    
                    new_score = current_score + next_score
                    new_seq = torch.cat([current_seq, torch.tensor([next_token], device=device)])
                    
                    if next_token == self.eos_token_id:
                        completed_beams[b_idx].append((new_score, new_seq))
                    else:
                        candidates[b_idx].append((new_score, new_seq))
            
            # Pruning
            for b_idx in range(batch_size):
                # Sort candidates by score (descending)
                candidates[b_idx].sort(key=lambda x: x[0], reverse=True)
                # Keep top beam_width
                active_beams[b_idx] = candidates[b_idx][:self.beam_width]
                
        # Final Gathering
        final_output = []
        for b_idx in range(batch_size):
            # Combine active and completed
            all_hyps = completed_beams[b_idx] + active_beams[b_idx]
            all_hyps.sort(key=lambda x: x[0], reverse=True)
            
            # Pad and stack
            # We want [beam_width, max_len]
            # If not enough hyps, pad with empty/pad tokens?
            # Or just repeat top?
            
            top_hyps = all_hyps[:self.beam_width]
            while len(top_hyps) < self.beam_width:
                 top_hyps.append(top_hyps[0] if top_hyps else (float('-inf'), torch.full((self.max_len,), self.pad_token_id, device=device)))
                 
            padded_hyps = []
            for score, seq in top_hyps:
                if len(seq) < self.max_len:
                    pad = torch.full((self.max_len - len(seq),), self.pad_token_id, device=device, dtype=torch.long)
                    seq = torch.cat([seq, pad])
                else:
                    seq = seq[:self.max_len]
                padded_hyps.append(seq)
                
            final_output.append(torch.stack(padded_hyps))
            
        return torch.stack(final_output) # [B, beam_width, max_len]
