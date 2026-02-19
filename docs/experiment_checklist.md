# RQ-VAE Experiment Checklist

## Pre-Experiment Setup

### 1. Environment Verification
```bash
# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should show: 2.1.1+cu121 (CUDA) or 2.1.1+rocm6.0 (AMD)

# For AMD: Verify env var
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### 2. Config Selection

**For Sliding Window (Recommended):**
```yaml
# configs/training/rqvae_sliding_window.yaml
rqvae:
  pooling_type: "sliding_window"
  window_size: 25      # ~0.5s
  window_stride: 12    # 50% overlap
  decay: 0.9           # Lower than 0.99 for faster updates
  commitment_cost: 0.25

training:
  batch_size: 32       # Reduce to 16 if OOM
  lr: 1e-4
  epochs: 50
```

**For Global Pooling (Baseline):**
```yaml
# configs/training/rqvae.yaml
rqvae:
  pooling_type: "global"
  decay: 0.9
  commitment_cost: 0.25
```

### 3. Monitor Configuration
```yaml
monitoring:
  enabled: true
  enable_codebook: true
  enable_ema: true
  enable_recon: true
  enable_training: true
  enable_alerts: true
  summary_interval: 500
  stop_on_critical: false    # Set to true if you want auto-stop
```

---

## During Training - Key Metrics to Watch

### Step 1-100 (Warmup)
- [ ] **Loss**: Should start ~1000, not NaN/Inf
- [ ] **Utilization**: May be low (0-20%), this is OK during warmup
- [ ] **EMA updates**: Should be incrementing (check logs)

### Step 100-500 (Initialization)
- [ ] **Utilization**: Should reach 15-30%
- [ ] **Loss**: Should start decreasing
- [ ] **VQ Loss**: Should be < 1.0
- [ ] **No critical alerts** (expected warnings are OK)

### Step 500 (First Checkpoint)
**Critical Decision Point:**

| Utilization | Action |
|-------------|--------|
| **> 30%** | ✅ Good! Continue training. Expected: 50-60% at step 1000 |
| **15-30%** | ⚠️ Marginal. Try reducing `decay` to 0.5 or increasing `window_size` to 50 |
| **< 15%** | ❌ Poor. Stop and try frame-level VQ or check data quality |

### Step 1000 (Second Checkpoint)
**Expected Targets:**
- [ ] **Utilization**: 40-60%
- [ ] **Loss**: < 600 (from ~1000)
- [ ] **Recon Loss**: Dominates total loss (>80%)
- [ ] **VQ Loss**: Stable, < 0.5

---

## Expected Results

### Healthy Training Pattern
```
Step 100:  Loss=950,  Util=15%, SNR=5dB   [WARMUP]
Step 500:  Loss=700,  Util=35%, SNR=10dB  [INITIALIZING]
Step 1000: Loss=500,  Util=50%, SNR=15dB  [LEARNING]
Step 5000: Loss=300,  Util=60%, SNR=20dB  [CONVERGED]
```

### Warning Signs

**If you see these, stop and adjust:**

1. **NaN/Inf loss immediately**
   - Cause: Learning rate too high or numerical instability
   - Fix: Reduce `lr` to 5e-5, increase warmup

2. **Utilization 0% after 500 steps**
   - Cause: Codebook not updating
   - Fix: Check `decay` is not 0.999, verify EMA updates

3. **Loss increasing**
   - Cause: Model diverging
   - Fix: Reduce `commitment_cost`, add gradient clipping

4. **VQ Loss >> Recon Loss**
   - Cause: VQ dominating training
   - Fix: Reduce `commitment_cost` to 0.1

---

## Post-Experiment Validation

### 1. Codebook Health Check
```bash
python scripts/phase0_prep/diagnose_rqvae.py \
    --checkpoint rqvae_checkpoint_5000.pt \
    --test-forward
```

### 2. Verify Encoding Works
```python
import torch
from speechgr.models.rqvae import SlidingWindowDocumentRQVAE

model = SlidingWindowDocumentRQVAE(...)
checkpoint = torch.load('rqvae_final.pt')
model.load_state_dict(checkpoint)
model.eval()

# Test encoding
dummy_audio = torch.randn(1, 500, 1024)  # [B, T, D]
codes = model.encode(dummy_audio)  # Should return [1, 8] for retrieval
print(f"Encoded codes: {codes.shape}")
print(f"Code range: [{codes.min()}, {codes.max()}]")
```

### 3. Check Reconstruction Quality
```python
# Load a real sample
features = ...  # From WavLM
codes = model.encode(features)

# Verify codes are diverse
unique_codes = len(torch.unique(codes))
print(f"Unique codes in sample: {unique_codes}")
# Should be > 5 for 8 codebooks
```

---

## Quick Commands

### Start Training
```bash
# Sliding window (recommended)
uv run python scripts/phase0_prep/train_rqvae.py \
    --config-path configs/training \
    --config-name rqvae_sliding_window \
    data.manifest_path=/path/to/train.txt \
    data.val_manifest=/path/to/val.txt

# For AMD GPU
export HSA_OVERRIDE_GFX_VERSION=11.0.0
./run_rocm.sh python scripts/phase0_prep/train_rqvae.py ...
```

### Monitor Training
```bash
# Watch wandb logs
# Check console output every 500 steps for summary

# Quick check utilization from checkpoint
python -c "
import torch
checkpoint = torch.load('rqvae_checkpoint_1000.pt')
# Check if model has trained
print('Checkpoint loaded successfully')
"
```

---

## Success Criteria

**Minimum Viable Model:**
- [ ] Utilization > 40% at step 5000
- [ ] Loss < 600 at step 5000
- [ ] No NaN/Inf during training
- [ ] Can encode and decode without errors

**Good Model:**
- [ ] Utilization > 50% at step 5000
- [ ] Loss < 400 at step 5000
- [ ] SNR > 15dB
- [ ] Stable VQ loss (< 0.5)

---

## If Things Go Wrong

### Utilization Stays Low (< 20%)

**Try these in order:**

1. **Increase window size**
   ```yaml
   window_size: 50  # Instead of 25
   window_stride: 25
   ```

2. **Lower EMA decay**
   ```yaml
   decay: 0.5  # Instead of 0.9
   ```

3. **Try frame-level VQ**
   - Modify code to skip pooling entirely
   - Each frame gets quantized independently
   - Mean pool for retrieval

4. **Check data quality**
   - Verify audio files are valid
   - Check SSL features are not all zeros/NaN

### Training Unstable

1. **Lower learning rate**
   ```yaml
   lr: 5e-5  # Instead of 1e-4
   ```

2. **Increase gradient clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Instead of 1.0
   ```

3. **Reduce commitment cost**
   ```yaml
   commitment_cost: 0.1  # Instead of 0.25
   ```

---

## Summary

**Best config to start:**
- Sliding window (rqvae_sliding_window.yaml)
- window_size=25, stride=12
- decay=0.9, commitment_cost=0.25
- lr=1e-4, batch_size=32
- stop_on_critical=false

**Watch for:**
- Utilization > 30% at step 500
- Loss trending down
- No NaN/Inf

**Decision at step 500:**
- Good (>30%): Continue to 5000 steps
- Marginal (15-30%): Try decay=0.5
- Poor (<15%): Check data or try frame-level

Good luck! 🚀
