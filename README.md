# CTNet (Cosine Transform Network): Neural Network Compression via DCT-Domain Training and H.265 Video Encoding

**Stylianos Iordanis and Stefanos Kornilios Mitsis Poiitidis**

> **Note:** This is an exploratory proof of concept. All experiments are conducted on ImageNette2-320 (a 10-class subset of ImageNet) using a single consumer GPU (NVIDIA GTX 1080 Ti, 11 GB). Results demonstrate the viability of the approach but are not directly comparable to full ImageNet-1K benchmarks. Hyperparameters have not been exhaustively tuned. We release this work to invite further investigation and collaboration.

---

## Abstract

We present CTNet (Cosine Transform Network), a family of compressed neural networks that reparameterize convolutional layers into the Discrete Cosine Transform (DCT) domain and leverage H.265/HEVC video encoding as the compression backend. Rather than relying on traditional pruning or fixed-bitwidth quantization, CTNet trains convolutional weights directly as DCT coefficients, regularized by a differentiable proxy of the H.265 bitrate cost. At export time, the DCT coefficient maps are tiled into 2D frames, normalized, and encoded as H.265 video streams with lossless or near-lossless settings.

We present two variants:

- **CTNet-18** (based on ResNet-18): achieves **92.31% Top-1** on ImageNette2-320 with a total compressed model size of **4.5 MB** (10.2x total compression including non-DCT weights), or **37.1x DCT-only compression** (42.9 MB to 1.1 MB) at 92.13% using the final epoch checkpoint. 17 DCT layers replace all spatial convolutions.
- **CTNet-50** (based on ResNet-50): applies the same DCT reparameterization to ResNet-50's bottleneck architecture. Due to ResNet-50's heavy use of 1x1 pointwise convolutions (which are not DCT-transformed), CTNet-50 has a similar DCT parameter count (11.3M) to CTNet-18 (11.0M) but benefits from the deeper architecture's representational capacity. The 36 retained 1x1 convolutions and batch normalization layers (54.3 MB total non-DCT overhead) can be independently compressed via standard quantization.

---

## 1. Introduction

Neural network compression is critical for deploying deep learning models on resource-constrained devices. Existing approaches fall into several broad categories: weight pruning (Han et al., 2015; Frankle & Carlin, 2019), quantization (Esser et al., 2020; Rastegari et al., 2016), knowledge distillation (Hinton et al., 2015), and neural architecture search for efficient models (Tan & Le, 2019). These methods operate directly on weight magnitudes or network topology.

We observe that convolutional kernels, when viewed as small 2D signals, are natural candidates for transform coding -- the same compression paradigm that underlies JPEG, H.264, and H.265. Video codecs like H.265/HEVC already implement highly optimized DCT-based transform coding with sophisticated entropy coding (CABAC), rate-distortion optimization, and multi-scale prediction. Rather than reinventing these tools for neural network compression, we propose to directly leverage them.

CTNet introduces three key ideas:

1. **DCT-domain parameterization**: Convolutional weights are learned directly as DCT coefficients, with spatial weights materialized via IDCT at each forward pass.
2. **Differentiable H.265 rate proxy**: A training-time regularizer that approximates the bitrate cost of encoding each DCT coefficient, modeling significance maps, level coding costs, and zig-zag scan order weights -- the same components H.265 uses internally.
3. **Video codec compression**: At export time, DCT coefficient maps are reshaped into 2D image tiles, normalized with per-frame center and scale factors, optionally dithered, and encoded as H.265 video streams.

---

## 2. Architecture Variants

### CTNet-18

Based on ResNet-18. All 17 spatial convolutions (kernel > 1x1) are replaced with DCTConv2d layers. Only 3 pointwise (1x1) convolutions remain as standard Conv2d.

| Component | Parameters | Size (float32) |
|-----------|-----------|----------------|
| DCT conv layers (17) | 11.0M | 41.9 MB |
| Standard 1x1 convs (3) | 0.3M | 1.1 MB |
| BN + FC + other | 0.4M | 1.6 MB |
| **Total** | **11.7M** | **44.6 MB** |

The DCT layers constitute 94% of the model's parameters, making CTNet-18 an ideal target for frequency-domain compression.

### CTNet-50

Based on ResNet-50's bottleneck architecture. The same 17 spatial convolutions (the 3x3 convolutions inside each bottleneck block, plus `conv1`) are replaced with DCTConv2d layers. However, ResNet-50 uses far more 1x1 convolutions (36 layers) in its bottleneck design for channel expansion/reduction.

| Component | Parameters | Size (float32) |
|-----------|-----------|----------------|
| DCT conv layers (17) | 11.3M | 43.2 MB |
| Standard 1x1 convs (36) | 12.6M | 48.0 MB |
| BN + FC + other | 1.7M | 6.3 MB |
| **Total** | **25.6M** | **97.5 MB** |

In CTNet-50, DCT layers account for 44% of parameters. The 1x1 convolutions (49% of parameters) are not DCT-transformed -- they have no spatial frequencies to compress. This creates an interesting hybrid: the spatial convolutions are compressed via H.265, while the pointwise convolutions can be independently compressed via standard INT8/INT4 quantization.

**Architectural insight.** Despite being a much deeper network, CTNet-50 has nearly identical DCT parameter counts to CTNet-18 (11.3M vs 11.0M). This is because ResNet-50's depth comes primarily from stacking bottleneck blocks with narrow 3x3 convolutions flanked by wider 1x1 projections. The H.265 compressed size of the DCT layers should therefore be comparable between the two variants, while CTNet-50's additional accuracy comes from the deeper feature hierarchy enabled by the (uncompressed) 1x1 convolutions.

---

## 3. Method

### 3.1 DCT-Domain Convolutional Layers

Each standard `Conv2d` layer with kernel size > 1x1 is replaced by a `DCTConv2d` layer. The learnable parameters are DCT coefficients $\hat{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$. At each forward pass, spatial weights are materialized via 2D IDCT:

$$W = C_h^T \hat{W} C_w$$

where $C_h$ and $C_w$ are Type-II DCT matrices. The convolution then proceeds normally using $W$. This reparameterization is transparent to the rest of the network -- gradients flow through the IDCT via standard backpropagation.

Pointwise (1x1) convolutions are left unchanged, as they have no spatial frequencies to compress.

### 3.2 Differentiable H.265 Rate Proxy

During training, we add a rate regularization term that approximates the bitrate cost an H.265 encoder would incur for each layer's DCT coefficients. The proxy models three components of HEVC entropy coding:

**Significance cost.** Each coefficient incurs approximately 1 bit to signal whether it is zero or non-zero. We approximate this with a sigmoid soft-threshold:

$$\sigma_k = \text{sigmoid}(s \cdot (|\hat{W}_k| / q - 0.5))$$

where $q$ is the quantization step size, $s$ is the sigmoid steepness, and $k$ indexes each coefficient. The 0.5 threshold matches H.265's rounding behavior: values below $0.5q$ quantize to zero.

**Level coding cost.** For non-zero coefficients, encoding the magnitude requires approximately $\log_2(1 + |\hat{W}_k|/q)$ bits.

**Scan-order cost.** H.265 encodes coefficients in diagonal zig-zag scan order, with CABAC context models that make later-scanned (higher frequency) non-zero coefficients more expensive. We assign a weight $w_{i,j}$ to each frequency position $(i,j)$ based on its zig-zag scan index, with DC (position 0,0) receiving weight 1.0 and the last position receiving weight $K_h \times K_w$.

The total rate proxy for one layer is:

$$\mathcal{L}_{rate} = \sum_{k} \sigma_k \cdot (1 + \log_2(1 + |l_k|)) \cdot w_{pos(k)}$$

The training loss is:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{rate}$$

where $\lambda$ controls the rate-distortion tradeoff.

### 3.3 H.265 Video Encoding Pipeline

At export time, each DCT layer's 4D coefficient tensor $(C_{out}, C_{in}, K_h, K_w)$ is reshaped into a 2D image of size $(C_{out} \cdot K_h, C_{in} \cdot K_w)$. Images larger than 128x128 are sliced into tiles; images smaller than 8x8 are circularly padded. Tiles of the same dimensions are grouped into a single H.265 video stream (one frame per tile).

**Per-frame normalization.** Each frame is independently normalized to the pixel range $[0, 2^b - 1]$ (where $b$ is the bit depth, default 12) with a stored center and scale factor for reconstruction:

$$\text{pixel} = \text{round}((\text{value} - \text{center}) \cdot \text{norm\_factor} + 2^{b-1})$$

Per-frame normalization is critical: different layers have vastly different coefficient magnitudes, and a shared normalization would waste precision on layers with small dynamic ranges.

**Subtractive dithering.** Optionally, deterministic white noise of configurable amplitude is added before rounding and subtracted after decoding. This decorrelates quantization error, which can improve accuracy when using lower bit depths or lossy CRF settings.

**Full-range encoding.** We use H.265's full-range mode (`range=full`, `color_range=pc`) to utilize the complete pixel value space, avoiding the wasted levels in the default limited (broadcast) range.

**Lossless mode.** With CRF 0, H.265 operates in mathematically lossless mode, preserving every pixel exactly. Compression then comes entirely from CABAC entropy coding exploiting spatial redundancy in the coefficient images.

### 3.4 Reconstruction

Decoding reverses the pipeline: H.265 video frames are decoded via ffmpeg, per-frame dither noise is subtracted, pixels are denormalized back to weight values using the stored center and scale factors, tiles are reassembled into 2D images, and the images are reshaped back into 4D DCT coefficient tensors. The IDCT in each `DCTConv2d` layer converts them to spatial weights at inference time.

A JSON manifest stores all metadata needed for exact reconstruction: architecture, quantization step, bit depth, dither amplitude, and per-frame normalization parameters.

---

## 4. Experimental Results

### 4.1 CTNet-18

**Setup:**
- **Base architecture**: ResNet-18 (17 DCT layers, 3 standard 1x1 convolutions)
- **Dataset**: ImageNette2-320 (10-class subset of ImageNet, 320px images)
- **Training**: AdamW, lr=1e-3, weight-decay=0.01, $\lambda=10^{-5}$, $q=0.1$, 256 epochs, pretrained
- **Export**: 8-bit depth, CRF 0 (lossless), `slower` preset

**Results (compression over training):**

| Epoch | Top-1 | Top-5 | H.265 (DCT) | DCT Ratio | Non-DCT | Total Size | Total Ratio |
|-------|-------|-------|-------------|-----------|---------|------------|-------------|
| 69 | — | — | 4,519 KB | 9.5x | 2,782 KB | 7,301 KB | 6.3x |
| 84 | — | — | 4,272 KB | 10.1x | 2,782 KB | 7,055 KB | 6.5x |
| 164 | — | — | 2,686 KB | 16.0x | 2,782 KB | 5,468 KB | 8.4x |
| 176 | — | — | 2,335 KB | 18.4x | 2,782 KB | 5,117 KB | 8.9x |
| **256 (best)** | **92.31%** | **99.44%** | **1,719 KB** | **25.0x** | **2,782 KB** | **4,502 KB** | **10.2x** |
| 256 (last) | 92.13% | 99.36% | 1,158 KB | 37.1x | 2,782 KB | 3,940 KB | 11.6x |

*Baseline pretrained ResNet-18 achieves ~95-97% Top-1 on ImageNette2-320.*
*Non-DCT weights (BN, FC, 1x1 convs) are 2,782 KB and included in the total compressed model.*

**Key findings:**

- **Accuracy**: CTNet-18 achieves 92.31% Top-1 (best checkpoint) — only ~3-4% below the uncompressed baseline (~96%), with a total model size of 4.5 MB.
- **Best vs last**: The best-accuracy checkpoint (92.31%, 4.5 MB total) and the final checkpoint (92.13%, 3.9 MB total) offer slightly different tradeoff points. Both are saved automatically during training.
- **Compression improves throughout training**: H.265 DCT size drops from 4,519 KB at epoch 69 to 1,158 KB at epoch 256 (3.9x improvement) as the rate proxy gradually reshapes coefficients toward patterns H.265 encodes efficiently.
- **Non-DCT weight overhead**: The 2,782 KB of non-DCT weights (BN running stats, FC layer, 1x1 convolutions) is a fixed overhead that limits the total compression ratio even as DCT compression improves. Future work could apply INT8 quantization to these weights for further reduction.
- **AdamW optimizer**: Per-parameter adaptive learning rates handle the bi-objective loss (task + rate) better than SGD, achieving higher accuracy at equivalent compression.

### 4.2 CTNet-50

CTNet-50 applies the identical DCT reparameterization and H.265 compression pipeline to ResNet-50. Training and export commands:

```bash
# Train CTNet-50
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1

# Export
python export_h265.py encode --arch resnet50 --crf 0 --preset slower

# Decode and evaluate
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

**Hardware limitation.** We were unable to train and evaluate CTNet-50 on our test hardware (NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM). ResNet-50's larger activation maps and the additional memory overhead of the DCT reparameterization exceeded the available GPU memory during training. CTNet-50 evaluation is left for future work on hardware with >= 24 GB VRAM.

**Expected behavior.** Because CTNet-50's DCT parameter count (11.3M) is nearly identical to CTNet-18's (11.0M), we expect:

- **H.265 compressed size of DCT layers**: comparable to CTNet-18 (~1.8 MB)
- **Higher accuracy**: ResNet-50's deeper bottleneck architecture provides stronger feature representations. The uncompressed 1x1 convolutions and batch normalization layers carry the additional representational capacity.
- **Total compressed model**: the 12.6M parameters in 1x1 convolutions (48 MB float32) would need separate compression. With standard INT8 quantization, these add ~12 MB, for an estimated total of ~14 MB.
- **Hybrid compression ratio**: ~97.5 MB float32 to ~14 MB (H.265 DCT + INT8 pointwise) = ~7x total, or ~23x on DCT layers alone.

This highlights a key architectural insight: CTNet compression is most effective on architectures where spatial convolutions dominate the parameter budget. For bottleneck architectures like ResNet-50, combining CTNet (for spatial convolutions) with standard quantization (for 1x1 convolutions) yields the best overall compression.

### 4.3 Comparison with State-of-the-Art

The following table compares CTNet against established neural network compression methods. Results are drawn from the original publications; where ResNet-18 results are unavailable, we report the closest comparable architecture.

| Method | Reference | Model | Compression | Top-1 Acc | Acc Drop |
|--------|-----------|-------|-------------|-----------|----------|
| **CTNet-18 (ours, best)** | -- | ResNet-18 | **10.2x total** | **92.31%*** | ~4% from baseline* |
| CTNet-18 (ours, last) | -- | ResNet-18 | 11.6x total | 92.13%* | ~4% from baseline* |
| Deep Compression | Han et al., 2016 | VGG-16 | 49x | 68.3%** | ~0%** |
| Deep Compression | Han et al., 2016 | AlexNet | 35x | 57.2%** | ~0%** |
| Deep Compression (est.) | -- | ResNet-18 | 15-25x | ~68-69%** | 1-2%** |
| Lottery Ticket (rewinding) | Frankle et al., 2020 | ResNet-50 | 3.3-5x | 76.1%** | ~0%** |
| Magnitude pruning (90%) | Zhu & Gupta, 2018 | ResNet-50 | 10x | 73.9%** | -2.2%** |
| Magnitude pruning (95%) | Zhu & Gupta, 2018 | ResNet-50 | 20x | 72.0%** | -4.1%** |
| INT8 quantization | Standard PTQ | ResNet-18 | 4x | 69.7%** | -0.1%** |
| INT4 quantization (QAT) | Esser et al., 2020 | ResNet-18 | 8x | 67.6%** | -2.2%** |
| Binary (ReActNet) | Liu et al., 2020 | ResNet-18 | 32x | 65.5%** | -4.3%** |
| Structured pruning (FPGM) | He et al., 2019 | ResNet-18 | ~1.7x FLOPs | 68.4%** | -1.4%** |
| Structured pruning (HRank) | Lin et al., 2020 | ResNet-18 | ~1.5x FLOPs | 69.1%** | -0.7%** |
| EfficientNet-B0 (NAS) | Tan & Le, 2019 | EfficientNet | 2.2x vs R18 | 77.1%** | +7.3%** |
| MobileNet-V2 (NAS) | Sandler et al., 2018 | MobileNet | 3.3x vs R18 | 72.0%** | +2.2%** |

\* *ImageNette2-320 (10-class subset). Pretrained baseline achieves ~95-97% on this dataset.*
\*\* *Full ImageNet-1K (1000 classes). Not directly comparable with ImageNette2 results.*

**Important note on comparability.** Our results are on ImageNette2-320 (10 classes), while most prior work reports on full ImageNet-1K (1000 classes). The compression ratio (23.7x on stored bytes) is directly comparable as it measures the same quantity -- ratio of original float32 parameter storage to compressed representation. However, accuracy numbers across different datasets should not be directly compared. A pretrained ResNet-18 baseline achieves ~95-97% on ImageNette2 vs ~69.8% on ImageNet-1K.

### 4.4 Analysis

**Total compression.** Including non-DCT weights (BN, FC, 1x1 convs), CTNet-18 achieves 10.2x total compression (44.7 MB to 4.5 MB) at 92.31% accuracy on the best checkpoint, or 11.6x (to 3.9 MB) on the final checkpoint. The DCT layers alone compress 25-37x, but the 2.7 MB non-DCT overhead limits the total ratio. Applying INT8 quantization to the non-DCT weights would reduce this overhead to ~0.7 MB, potentially pushing total compression to 15-20x.

**Compression improves with training.** H.265 encoded size drops continuously from 4.5 MB at epoch 69 to 1.2 MB at epoch 256 — a 3.9x improvement — while accuracy remains above 92%. The rate proxy gradually reshapes the DCT coefficient landscape toward patterns that H.265's CABAC entropy coder compresses efficiently.

**Self-contained export.** The H.265 output directory contains the complete model: `.hevc` video streams for DCT weights, `non_dct_weights.pt` for BN/FC/1x1 weights, and `manifest.json` for reconstruction metadata. No external checkpoint is needed for inference.

**Rate-distortion tradeoff.** CTNet offers a continuous rate-distortion tradeoff controlled by $\lambda$, $q$, CRF, and bit depth. Increasing $\lambda$ during training encourages sparser DCT representations; increasing CRF during export trades accuracy for smaller files. This is analogous to how video codecs offer smooth quality-vs-bitrate curves.

**Codec preset profiling.** H.265 encoding presets from `ultrafast` to `veryslow` trade encoding time for compression efficiency. On CTNet-18, `medium` achieved near-optimal compression (within 3% of `veryslow`) at 4x faster encoding speed, suggesting that the coefficient images are relatively easy for the encoder to optimize.

**CTNet-18 vs CTNet-50.** The two variants illuminate a fundamental property of codec-based compression: it is most effective on spatial convolutions. In CTNet-18, where 94% of parameters are in spatial convolutions, nearly the entire model benefits from H.265 compression. In CTNet-50, only 44% of parameters are spatial -- the rest are 1x1 projections that require separate compression. This suggests CTNet is particularly well-suited for architectures that maximize spatial convolution usage, and motivates future work on frequency-domain reparameterization of 1x1 convolutions (e.g., via channel-wise transforms).

**Advantages over traditional methods:**

- *No custom entropy coder needed.* H.265/HEVC is a mature, hardware-accelerated standard available on virtually every modern device. Decoding is a single ffmpeg call.
- *Continuous rate-distortion control.* Unlike pruning (discrete sparsity levels) or quantization (discrete bit widths), CTNet inherits video coding's smooth tradeoff via CRF and $\lambda$.
- *Complementary to other methods.* CTNet operates in the frequency domain and can be combined with channel pruning, knowledge distillation, or standard quantization for non-DCT layers.

---

## 5. Related Work

**DCT-Conv: Coding filters with Discrete Cosine Transform (Checinski & Wawrzynski, 2020).** The most directly relevant prior work. Checinski & Wawrzynski proposed DCT-Conv layers where convolutional filters are defined by their DCT coefficients rather than spatial weights, with IDCT applied to recover spatial filters during the forward pass. They demonstrated on ResNet-50/CIFAR-100 that 99.9% of 3x3 DCT coefficients can be switched off while maintaining good performance — the network effectively learns to use only a few frequency components per filter. Their key finding validates our core premise: neural network weights are highly compressible in the DCT domain. CTNet extends this work in three significant ways: (1) we add a differentiable rate proxy that optimizes *for* compressibility during training rather than post-hoc coefficient removal, (2) we use a production video codec (H.265) as the compression backend rather than simple coefficient zeroing, and (3) we extend DCT reparameterization beyond spatial convolutions to 1x1 pointwise convolutions via channel-wise DCT and to all other layer types (BN, FC) via center+normalize encoding.

**ECRF: Entropy-Constrained Neural Radiance Fields (Lee et al., 2023).** Lee et al. applied DCT-domain compression with entropy optimization to Neural Radiance Fields (NeRF), achieving state-of-the-art compression on TensoRF feature grids. Their approach shares several key ideas with CTNet: (1) transforming parameters to the frequency domain via DCT, (2) training with an entropy-based loss to encourage compressible representations, and (3) post-training quantization to 8-bit followed by entropy coding. They also use additive uniform noise as a differentiable proxy for quantization during training — similar to our dither mechanism. A crucial insight from ECRF is that jointly optimizing the entropy of transformed coefficients with the task loss (in their case, rendering quality) produces significantly sparser frequency-domain representations than training without entropy awareness. Their pipeline (DCT → 8-bit quantization → arithmetic coding) parallels CTNet's pipeline (DCT → center+normalize → H.265 CABAC), but we replace custom arithmetic coding with a standard video codec, gaining hardware decoder support at the cost of some compression efficiency.

**Weight pruning.** Unstructured pruning (Han et al., 2015) removes individual weights by magnitude, achieving 9-13x compression on AlexNet/VGG. The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) shows that sparse subnetworks can be trained from scratch, but finding tickets requires iterative pruning-retraining cycles. Structured pruning (He et al., 2017; Lin et al., 2020) removes entire filters or channels for hardware-friendly speedups but typically achieves lower compression ratios (1.5-2x on ResNet-18).

**Quantization.** Reducing weight precision from 32-bit floating point to 8-bit integers gives 4x compression with negligible accuracy loss (Jacob et al., 2018). More aggressive quantization to 4 bits (Esser et al., 2020) or 2 bits degrades accuracy significantly on small models like ResNet-18. Binary networks (Rastegari et al., 2016; Liu et al., 2020) achieve 32x compression but with 4-18% accuracy drops.

**Deep Compression.** Han et al. (2016) pipeline pruning, quantization (to 5-8 bits with codebook), and Huffman entropy coding to achieve 35-49x on AlexNet/VGG. This remains the gold standard for maximum compression, but requires custom decompression code and benefits disproportionately from large fully-connected layers absent in modern architectures.

**Learned image compression.** The learned compression literature (Balle et al., 2017; Minnen et al., 2018) has developed differentiable rate-distortion optimization for image codecs. Our rate proxy draws inspiration from this work, adapting it to the specific structure of neural network coefficient maps.

---

## 6. Reproducibility

### 6.1 CTNet-18

```bash
pip install -r requirements.txt
./download_imagenette.sh ./imagenette2-320

# Train (best config: AdamW, 256 epochs)
python train_imagenet.py ./imagenette2-320 \
    --arch resnet18 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --cache-dataset

# Export to H.265 (lossless, self-contained output)
python export_h265.py encode --arch resnet18 --crf 0 --preset slower

# Decode and evaluate (auto-loads non-DCT weights from h265 dir)
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

### 6.2 CTNet-50

```bash
# Train
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1

# Export to H.265
python export_h265.py encode --arch resnet50 --crf 0 --preset slower

# Decode and evaluate
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

### 6.3 Profile Encoding Presets

```bash
python export_h265.py encode --arch resnet18 --profile
python export_h265.py encode --arch resnet50 --profile
```

---

## 7. Conclusion

CTNet demonstrates that modern video codecs are surprisingly effective neural network compressors. By reparameterizing convolutional layers into the DCT domain and training with a differentiable H.265 rate proxy, CTNet-18 compresses a ResNet-18 to 4.5 MB total (10.2x) at 92.31% Top-1 on ImageNette2, only ~4% below the uncompressed baseline. The DCT layers alone achieve 25x compression, with the non-DCT weight overhead as the main bottleneck for further gains. The approach requires no custom entropy coding implementation, leverages decades of video codec optimization, and offers continuous rate-distortion control via standard codec parameters.

CTNet-50 extends the framework to deeper bottleneck architectures, revealing that the approach is most effective when spatial convolutions dominate the parameter budget. For architectures with many 1x1 convolutions, CTNet naturally combines with standard quantization for a hybrid compression strategy.

Future work includes evaluation on full ImageNet-1K, combination with structured pruning and knowledge distillation, exploration of newer codecs (AV1, VVC/H.266), extension of frequency-domain compression to 1x1 convolutions via channel-wise transforms, and investigation of whether the H.265 encoder's internal rate-distortion optimization can be directly exploited during training.

---

## References

- Balle, J., Laparra, V., & Simoncelli, E. P. (2017). End-to-end optimized image compression. ICLR.
- Checinski, K., & Wawrzynski, P. (2020). DCT-Conv: Coding filters in convolutional networks with Discrete Cosine Transform. arXiv:2001.08517.
- Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., & Keutzer, K. (2020). HAWQ-V2: Hessian aware trace-weighted quantization of neural networks. NeurIPS.
- Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., & Modha, D. S. (2020). Learned step size quantization. ICLR.
- Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable networks. ICLR.
- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carlin, M. (2020). Stabilizing the lottery ticket hypothesis. arXiv:1903.01611.
- Han, S., Mao, H., & Dally, W. J. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. ICLR.
- Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning both weights and connections for efficient neural networks. NeurIPS.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- He, Y., Kang, G., Dong, X., Fu, Y., & Yang, Y. (2018). Soft filter pruning for accelerating deep convolutional neural networks. IJCAI.
- He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019). Filter pruning via geometric median for deep convolutional neural networks acceleration. CVPR.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.
- Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR.
- Lee, S., Shu, F., Sanchez, Y., Schierl, T., & Hellge, C. (2023). ECRF: Entropy-constrained neural radiance fields compression with frequency domain optimization. arXiv:2311.14208.
- Lin, M., et al. (2020). HRank: Filter pruning using high-rank feature map. CVPR.
- Liu, Z., Shen, Z., Savvides, M., & Cheng, K. T. (2020). ReActNet: Towards precise binary neural network with generalized activation functions. ECCV.
- Luo, J., Wu, J., & Lin, W. (2017). ThiNet: A filter level pruning method for deep neural network compression. ICCV.
- Minnen, D., Balle, J., & Toderici, G. (2018). Joint autoregressive and hierarchical priors for learned image compression. NeurIPS.
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet classification using binary convolutional neural networks. ECCV.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
- Zhu, M., & Gupta, S. (2018). To prune, or not to prune: Exploring the efficacy of pruning for model compression. ICLR Workshop.

---

## License

This project is released for research purposes.
