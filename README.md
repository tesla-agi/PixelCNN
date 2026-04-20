# PixelCNN — Autoregressive Image Generation on MNIST

Two implementations of autoregressive image models trained on binarized MNIST, built from scratch in PyTorch. Both model `p(x) = ∏ p(xᵢ | x_{<i})` in raster scan order using masked convolutions.

## Files

- `pixelcnn.py` — Plain PixelCNN (van den Oord et al., 2016)
- `gated_pixelcnn.py` — Gated PixelCNN with two-stack architecture (van den Oord et al., 2016b)

---

## 1. Plain PixelCNN

An autoregressive image model that factors `p(x) = ∏ p(xᵢ | x_{<i})` in raster order. It uses masked convolutions (Mask A in layer 1, Mask B after) to ensure each pixel only sees previous ones. Training is one forward pass with BCE loss; sampling is sequential, one pixel at a time. Has a blind spot in the upper-right region.

### Architecture

- **Mask A** (first layer): blocks center pixel and all future pixels
- **Mask B** (all following layers): allows center pixel, blocks future
- 1 × MaskedConv (7×7, Mask A) → 7 × MaskedConv (3×3, Mask B) with ReLU → 1×1 output conv → sigmoid
- ReLU activations throughout

### Drawbacks

- **Blind spot**: upper-right context never reaches the current pixel because of how masked convs stack
- **Limited depth**: training becomes unstable beyond 7-8 layers due to vanishing gradients (no residual connections)
- **ReLU is binary**: hard on/off gating loses information

---

## 2. Gated PixelCNN

Gated PixelCNN fixes the blind spot by splitting computation into two parallel streams — a vertical stack that sees all rows above (full width) and a horizontal stack that sees current-row left context. Vertical features are injected into the horizontal stream at every layer. ReLU is replaced with gated activation `tanh(a) * sigmoid(b)`, borrowed from LSTM-style gating. Residual connections on the horizontal stack allow deeper networks. Same autoregressive guarantee, sharper samples.

### Architecture

**Gated activation** replaces ReLU:
```
gate(x) = tanh(first_half(x)) * sigmoid(second_half(x))
```
tanh controls the value, sigmoid controls how much flows through.

**Two masked convs**:
- `MaskedConv1` (vertical): allows only rows strictly above center
- `MaskedConv2` (horizontal): allows current row, left of center + center

**GatedLayer** (the core block):
1. Vertical conv → pre-gated features `v_feature`
2. Gate → `v_out` (flows to next layer's vertical input)
3. Horizontal conv → `h_feature`
4. Project `v_feature` via 1×1 conv, add to `h_feature`, gate → `h_activation`
5. 1×1 conv on `h_activation` + residual from `h_input` → `h_out`
6. Return both `v_out, h_out`

**Full model**:
- Vertical init: MaskedConv1 (1 → 2C, 7×7) + gated activation
- Horizontal init: MaskedConv2 (1 → C, 3×3), no gating
- 7 × GatedLayer(C=64)
- Output head: 1×1 conv → ReLU → 1×1 conv → sigmoid

### Why it's better

| | Plain PixelCNN | Gated PixelCNN |
|---|---|---|
| Blind spot | Yes — upper-right poorly covered | No — vertical stack sees full width |
| Activation | ReLU (binary on/off) | tanh ⊙ sigmoid (continuous control) |
| Residual connections | No | Yes (horizontal stack) |
| Sample quality | Blurry, noisy digits | Sharp, clear digits |
| Parameters | ~300k | ~1.2M |

---

## Training

Both models train with the same setup:
- Dataset: MNIST, binarized at threshold 0.5
- Loss: binary cross-entropy (pixel-wise)
- Optimizer: Adam, lr=0.001
- Batch size: 128
- Epochs: 20 (3 for quick testing)

## Sampling

Sampling is inherently sequential — 784 forward passes for a 28×28 image:
```python
for i in range(784):
    out = model(x)
    row, col = i // 28, i % 28
    x[:, 0, row, col] = torch.bernoulli(out[:, 0, row, col])
```
Cannot be parallelized because pixel `i+1` depends on pixel `i`. Same limitation as GPT generating text token-by-token.

## Performance Notes (CPU)

- Plain PixelCNN: ~1-2 min/epoch
- Gated PixelCNN: ~3-5 min/epoch (2x streams + more params)
- Sampling: 30s to 2 min per batch of 16
- On GPU both run ~10-50x faster

## Key Concepts Learned

- Chain rule factorization for joint distributions
- Masked convolutions for enforcing autoregressive ordering
- Why Mask A vs Mask B (center pixel treatment)
- How the blind spot arises in stacked masked convolutions
- Gated activations vs ReLU (LSTM-style gating for conv nets)
- Two-stack architecture for full autoregressive context
- Sequential sampling as a fundamental limitation of autoregressive models

## References

- Germain et al., 2015 — MADE: Masked Autoencoder for Distribution Estimation
- van den Oord et al., 2016 — Pixel Recurrent Neural Networks
- van den Oord et al., 2016 — Conditional Image Generation with PixelCNN Decoders (Gated PixelCNN)
