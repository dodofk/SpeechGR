import torch
import numpy as np
from unittest.mock import MagicMock, patch
from speechgr.encoders.rqvae.encoder import RQVAEEncoder

def test_rqvae_encoder():
    # 1. Setup Mocks
    with patch("speechgr.models.ssl_wrapper.AutoModel") as mock_auto_model, \
         patch("speechgr.models.ssl_wrapper.AutoConfig") as mock_auto_config, \
         patch("speechgr.encoders.rqvae.encoder.torch.load") as mock_torch_load:
         
        # Mock SSL
        mock_config_inst = MagicMock()
        mock_config_inst.hidden_size = 768
        mock_auto_config.from_pretrained.return_value = mock_config_inst
        
        mock_ssl_model = MagicMock()
        # Mock forward to return a TENSOR, not a Mock, so downstream ops work
        mock_ssl_model.return_value.hidden_states = [torch.randn(1, 50, 768) for _ in range(13)]
        mock_auto_model.from_pretrained.return_value = mock_ssl_model
        
        # Mock Checkpoint
        mock_torch_load.return_value = {} 
        
        # Initialize Encoder
        # Default behavior is document-level RQ-VAE.
        with patch("speechgr.encoders.rqvae.encoder.DocumentRQVAE") as MockRQVAE:
            mock_rqvae_inst = MockRQVAE.return_value
            # Handle .to() fluent interface
            mock_rqvae_inst.to.return_value = mock_rqvae_inst
            
            # Mock encode: returns real Tensor [1, 4] for document-level codes.
            # This allows .squeeze().cpu().numpy() to work for real
            mock_rqvae_inst.encode.return_value = torch.randint(0, 1024, (1, 4))
            
            encoder = RQVAEEncoder(
                ssl_model_name="dummy",
                rqvae_checkpoint="dummy.pt",
                rqvae_config={"codebook_size": 1024, "num_codebooks": 4}
            )
            
            # 2. Test encode_audio
            dummy_audio = np.random.randn(16000)
            codes = encoder.encode_audio(dummy_audio, 16000)
            
            # Expected output: num_codebooks = 4
            print(f"Codes shape: {codes.shape}")
            assert codes.shape == (4,)
            assert codes.dtype == np.int64
            
            print("RQVAEEncoder test passed!")

if __name__ == "__main__":
    test_rqvae_encoder()
