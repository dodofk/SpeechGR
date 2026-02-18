import torch
import json
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from speechgr.models.unity import UnitySpeechModel
from speechgr.data.dual_task import DualTaskDataset, DualTaskCollator
from speechgr.utils.trie import Trie, TrieLogitsProcessor

def calculate_recall(preds, targets, k_list=[1, 5, 10]):
    """
    preds: List of List of IDs (each inner list is a hypothesis)
           Wait, preds should be List of [Top-K Sequences]
    targets: List of single ground truth IDs
    """
    recalls = {k: 0 for k in k_list}
    for candidates, target in zip(preds, targets):
        # candidates is a list of sequences e.g. [[1,2,3], [4,5,6], ...]
        # target is a sequence [1,2,3]
        
        # Check if target is in top-K candidates
        for k in k_list:
            top_k_hyps = candidates[:k]
            # Exact match check
            if target in top_k_hyps:
                recalls[k] += 1
    
    return {k: v / len(targets) for k, v in recalls.items()}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = DualTaskDataset(
        id_map_path=args.id_map,
        semantic_map_path=args.semantic_map,
        audio_root=args.audio_root,
        is_training=False # Use full doc or test queries
    )
    
    collator = DualTaskCollator()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
    
    # 2. Build Trie
    print("Building Trie from ID Map...")
    with open(args.id_map, 'r') as f:
        id_map = json.load(f)
    trie = Trie(id_map.values())
    trie_processor = TrieLogitsProcessor(trie, start_token_id=0)
    
    # 3. Load Model
    print(f"Loading Model from {args.checkpoint}...")
    model = UnitySpeechModel(
        ssl_model_name=args.ssl_model,
        semantic_vocab_size=args.semantic_vocab_size,
        retrieval_vocab_size=args.retrieval_vocab_size,
        d_model=args.d_model
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # 4. Inference
    all_preds = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # For Recall@K, we ideally need beam search
            # UnitySpeechModel.generate currently does greedy if num_beams=1
            # We use num_beams > 1 for Recall calculation
            _, ret_gen = model.generate(
                input_values=batch["input_values"],
                attention_mask=batch["attention_mask"],
                max_ret_len=8,
                logits_processor=trie_processor,
                num_beams=args.num_beams
            )
            
            # ret_gen: [B, num_beams, L] if beam search
            # We want the top-1 prediction for now, which is beam index 0
            # Or if we want to support Recall@K with multiple beams, we need to pass all beams.
            
            # Extract top beam
            if ret_gen.dim() == 3:
                # [B, num_beams, L]
                # Top beam is at index 0 because beam search sorts them
                top_beams = ret_gen[:, 0, :] # [B, L]
                
                # If we want to evaluate Recall with multiple hypotheses, we should collect all beams
                # all_preds structure: List of (List of IDs) or (List of List of IDs)
                # calculate_recall expects preds to be List of Top-K candidates
                
                # Let's pass ALL beams to calculate_recall
                # Remove BOS from all beams
                batch_preds = []
                for b in range(ret_gen.size(0)):
                    beams = []
                    for k in range(ret_gen.size(1)):
                        # Remove BOS (index 0)
                        seq = ret_gen[b, k, 1:].cpu().tolist()
                        beams.append(seq)
                    batch_preds.append(beams)
                
                all_preds.extend(batch_preds)
                
            else:
                # Greedy: [B, L]
                pred_id = ret_gen[:, 1:].cpu().tolist()
                # Wrap in list to match structure [[pred]]
                all_preds.extend([[p] for p in pred_id])
            
            target_id = batch["retrieval_labels"][:, 1:].cpu().tolist()
            all_targets.extend(target_id)
            
    # Flatten batches
    # Since we only have top-1 for now, Recall@5/10 will be same as Recall@1
    results = calculate_recall(all_preds, all_targets)
    print("\nResults:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_map", type=str, required=True)
    parser.add_argument("--semantic_map", type=str, required=True)
    parser.add_argument("--audio_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ssl_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--semantic_vocab_size", type=int, default=5000)
    parser.add_argument("--retrieval_vocab_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=768)
    
    args = parser.parse_args()
    main(args)
