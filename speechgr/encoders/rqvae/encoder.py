import torch
import numpy as np
from typing import Any, Dict, Iterable, Mapping, Optional
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from speechgr.encoders.base import ModalityEncoder
from speechgr.models.ssl_wrapper import SSLModelWrapper
from speechgr.models.rqvae import RQVAE, DocumentRQVAE

class RQVAEEncoder(ModalityEncoder):
    """
    Encoder that uses a pre-trained RQ-VAE model to discretize audio.
    Pipeline: Audio -> SSL Model -> Features -> RQ-VAE -> Codes.
    """
    def __init__(
        self,
        *,
        ssl_model_name: str,
        rqvae_checkpoint: str,
        ssl_layer: int = -1,
        rqvae_config: Dict[str, Any] = None,
        audio_field: str = "audio",
        sample_id_field: str = "id",
        is_document_level: bool = True,
        cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(name="rqvae", cfg=cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.audio_field = audio_field
        self.sample_id_field = sample_id_field
        self.is_document_level = is_document_level
        
        # Load SSL Model
        self.ssl_model = SSLModelWrapper(
            model_name=ssl_model_name,
            layer=ssl_layer,
            freeze=True
        ).to(self.device)
        self.ssl_model.eval()
        
        # Load RQ-VAE
        if rqvae_config is None:
             rqvae_config = {
                 "latent_dim": self.ssl_model.feature_dim,
                 "codebook_size": 256,
                 "num_codebooks": 8,
                 "commitment_cost": 0.25
             }
        
        if is_document_level:
            self.rqvae = DocumentRQVAE(
                input_dim=self.ssl_model.feature_dim,
                latent_dim=rqvae_config.get("latent_dim", self.ssl_model.feature_dim),
                codebook_size=rqvae_config.get("codebook_size", 256),
                num_codebooks=rqvae_config.get("num_codebooks", 8),
                commitment_cost=rqvae_config.get("commitment_cost", 0.25)
            ).to(self.device)
        else:
            self.rqvae = RQVAE(
                input_dim=self.ssl_model.feature_dim,
                latent_dim=rqvae_config.get("latent_dim", self.ssl_model.feature_dim),
                codebook_size=rqvae_config.get("codebook_size", 1024),
                num_codebooks=rqvae_config.get("num_codebooks", 4),
                commitment_cost=rqvae_config.get("commitment_cost", 0.25)
            ).to(self.device)
        
        # Load state dict
        checkpoint = torch.load(rqvae_checkpoint, map_location=self.device)
        self.rqvae.load_state_dict(checkpoint)
        self.rqvae.eval()
        
        self._supports_precompute = True

    def supports_precompute(self) -> bool:
        return self._supports_precompute

    def encode_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Encodes audio to codes.
        Returns: 
            If document-level: [D] array.
            If sequence-level: [T * D] array.
        """
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 2. Extract SSL Features
            features = self.ssl_model(audio_tensor) # [1, T, D_ssl]
            
            # 3. RQ-VAE Encoding
            codes = self.rqvae.encode(features) # [1, D] or [1, T, D]
            
        # 4. Flatten: [1, D] -> [D] or [1, T, D] -> [T*D]
        codes = codes.squeeze(0).cpu().numpy()
        flattened_codes = codes.flatten()
        
        return flattened_codes.astype(np.int64)

    def precompute(
        self,
        dataset_split: str,
        output_dir: str,
        samples: Iterable[Mapping[str, Any]],
    ) -> None:
        cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for sample in samples:
            if self.audio_field not in sample:
                raise KeyError(f"Expected audio field '{self.audio_field}'")
            
            audio_entry = sample[self.audio_field]
            audio = audio_entry["array"]
            sampling_rate = audio_entry["sampling_rate"]
            
            sample_id = sample.get(self.sample_id_field)
            if sample_id is None:
                raise KeyError(f"Expected id field '{self.sample_id_field}'")
                
            codes = self.encode_audio(np.asarray(audio), int(sampling_rate))
            cache[str(sample_id)] = {
                "codes": torch.tensor(codes, dtype=torch.long)
            }
            
        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def load_feature(
        self, dataset_split: str, output_dir: str, sample_id: str
    ) -> Dict[str, torch.Tensor]:
        return super().load_feature(dataset_split, output_dir, sample_id)

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        loaded = torch.load(path, map_location="cpu")
        return {str(k): v for k, v in loaded.items()}

__all__ = ["RQVAEEncoder"]
