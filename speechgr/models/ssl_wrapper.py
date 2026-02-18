import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class SSLModelWrapper(nn.Module):
    """
    Wrapper for HuggingFace Transformers audio models (HuBERT, WavLM, etc.).
    Extracts features from a specified layer.
    """
    def __init__(
        self, 
        model_name: str, 
        layer: int = -1, 
        freeze: bool = True,
        output_dim: int = None
    ):
        """
        Args:
            model_name: Name of the model to load via transformers.AutoModel.
            layer: Index of the hidden state to use. -1 for the last hidden state.
            freeze: Whether to freeze the model parameters.
            output_dim: Optional expected output dimension. If provided, checks consistency.
        """
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        
        # Load config first to check parameters if needed
        self.config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        if output_dim is not None and self.hidden_size != output_dim:
            raise ValueError(
                f"Model {model_name} has hidden size {self.hidden_size}, "
                f"but expected {output_dim}"
            )
            
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: Audio input (B, T)
            attention_mask: Mask (B, T)
            
        Returns:
            features: (B, T_subsampled, D)
        """
        outputs = self.model(
            input_values, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # hidden_states is a tuple of (initial_embeddings, layer_1, ..., layer_N)
        # So index 0 is embeddings, index 1 is layer 1, etc.
        # But transformers `output_hidden_states=True` usually returns all layers.
        # Let's handle generic indexing.
        
        all_states = outputs.hidden_states
        
        # Simple indexing
        try:
            features = all_states[self.layer]
        except IndexError:
            # Handle standard python negative indexing logic manually if needed 
            # or usually python list supports negative index.
            # But all_states is a tuple.
            if self.layer < -len(all_states) or self.layer >= len(all_states):
                 raise ValueError(f"Layer {self.layer} out of bounds for {len(all_states)} layers")
            features = all_states[self.layer]
            
        return features

    @property
    def feature_dim(self):
        return self.hidden_size
