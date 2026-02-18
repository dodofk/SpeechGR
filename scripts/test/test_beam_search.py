import torch
from speechgr.models.unity import UnitySpeechModel

def test_beam_search():
    device = torch.device("cpu")
    print("Initializing Model...")
    model = UnitySpeechModel(
        ssl_model_name="hf-internal-testing/tiny-random-wavlm",
        ssl_layer=2,
        d_model=32,
        num_heads=4,
        num_layers=2
    ).to(device)
    
    # Dummy input
    B = 2
    T = 16000
    input_values = torch.randn(B, T).to(device)
    
    print("Testing Greedy Generation (num_beams=1)...")
    sem, ret_greedy = model.generate(input_values, num_beams=1)
    print(f"Greedy Output Shape: {ret_greedy.shape}")
    assert ret_greedy.shape == (B, 9), f"Expected (B, 9), got {ret_greedy.shape}"
    
    print("Testing Beam Search (num_beams=3)...")
    sem, ret_beam = model.generate(input_values, num_beams=3)
    print(f"Beam Output Shape: {ret_beam.shape}")
    
    # Expected output shape for beam search: [B, num_beams, L]
    # Current implementation returns [B, num_beams, L]
    # L should be 9 (BOS + 8 generated)
    
    assert ret_beam.shape == (B, 3, 9), f"Expected (B, 3, 9), got {ret_beam.shape}"
    
    print("Beam Search Test Passed!")

if __name__ == "__main__":
    test_beam_search()
