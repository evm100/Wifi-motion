# Module 4: Advanced Processing & Deep Learning
## RTX 4080 — Model Architectures, Multi-Node Fusion & Domain Adaptation

---

## 1. From CSI Tensors to Classification

The RTX 4080 receives preprocessed tensors from the Raspberry Pi (Module 3) and must solve three increasingly difficult problems:

1. **Activity Recognition** — What is the person doing? (walking, sitting, falling, gesturing)
2. **Localization** — Where is the person in the room?
3. **Domain Adaptation** — Can the model work in a new room without retraining?

Each problem demands different data representations and model architectures. This module covers all three.

---

## 2. Data Representations for Deep Learning

### 2.1 The Four Input Formats

Raw CSI can be transformed into several formats for neural networks, each with tradeoffs:

| Representation | Shape | Best For | Model Type |
|---------------|-------|----------|------------|
| **Raw CSI amplitude** | [T, S, N] | Simple presence detection | 1D-CNN, LSTM |
| **Doppler spectrogram** | [S, F, T'] | Activity recognition | 2D-CNN (image) |
| **Body-coordinate Velocity Profile (BVP)** | [F, 2, T'] | Cross-domain gestures | CNN-GRU hybrid |
| **PCA score time series** | [T, K] | Real-time streaming | LSTM, Transformer |

Where T = time samples, S = subcarriers, N = nodes, F = frequency bins, T' = spectrogram time bins, K = PCA components.

### 2.2 Building CSI Spectrograms (Primary Approach)

The Doppler spectrogram from Module 3 is the dominant input format in the literature. For your multi-node system, you construct a **multi-channel spectrogram image** where each channel represents a different spatial perspective:

```python
import torch
import numpy as np
from scipy.signal import stft

def build_spectrogram_tensor(csi_window, fs=100.0, nperseg=128,
                              noverlap=120, n_pcs=10):
    """
    Convert multi-node CSI window into a GPU-ready spectrogram tensor.

    Args:
        csi_window: dict {node_id: np.array[T, n_subcarriers] complex CSI}
        fs: Sampling rate (Hz)
        nperseg: STFT window length
        noverlap: STFT overlap
        n_pcs: Number of PCA components to use per node

    Returns:
        tensor: torch.Tensor [C, H, W] where
                C = n_nodes * n_pcs (channel dimension)
                H = frequency bins
                W = time bins
    """
    channels = []

    for node_id in sorted(csi_window.keys()):
        csi = csi_window[node_id]  # [T, S] complex

        # Use amplitude of top PCA components (or selected subcarriers)
        amplitude = np.abs(csi)

        # Compute spectrogram per subcarrier (or per PCA component)
        for sc_idx in range(min(n_pcs, amplitude.shape[1])):
            f, t, Zxx = stft(amplitude[:, sc_idx], fs=fs,
                            nperseg=nperseg, noverlap=noverlap)
            power_db = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-10)
            channels.append(power_db)

    # Stack into [C, H, W] tensor
    tensor = np.stack(channels, axis=0)  # [C, freq, time]

    # Normalize to [0, 1] for CNN input
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-10)

    return torch.FloatTensor(tensor)
```

### 2.3 Body-coordinate Velocity Profile (BVP)

BVP is the key innovation from Widar3.0 that enables domain-independent gesture recognition. It transforms environment-dependent Doppler shifts into body-coordinate velocities by modeling the geometric relationship between the person, TX, and each RX.

For a person at position p with velocity v, the Doppler shift observed at link (TX, RX_i) is:

```
f_D,i = (v · (d_TX + d_RX_i)) / (λ · |d_TX| · |d_RX_i|)
```

Where d_TX and d_RX_i are unit vectors from the person to the TX and RX_i. The BVP inverts this: given Doppler shifts from 3+ links, solve for the velocity vector v in the person's body coordinate frame. This velocity profile is independent of room geometry, furniture, and device placement.

```python
def estimate_bvp(doppler_profiles, tx_pos, rx_positions, person_pos,
                  person_orientation, wavelength=0.125):
    """
    Estimate Body-coordinate Velocity Profile from multi-link Doppler.

    This is a simplified version — full Widar3.0 uses compressed sensing.

    Args:
        doppler_profiles: dict {node_id: [F, T'] Doppler power spectra}
        tx_pos: [x, y] TX position in room coordinates
        rx_positions: dict {node_id: [x, y]} RX positions
        person_pos: [x, y] estimated person position (from localization)
        person_orientation: float, body orientation angle (radians)
        wavelength: WiFi wavelength (0.125m for 2.4GHz)

    Returns:
        bvp: [2, T'] body-coordinate velocity profile (vx, vy)
    """
    n_links = len(rx_positions)

    # Compute direction vectors from person to TX and each RX
    d_tx = np.array(tx_pos) - np.array(person_pos)
    d_tx_norm = d_tx / (np.linalg.norm(d_tx) + 1e-10)

    # Build projection matrix A where each row maps velocity to Doppler
    A = np.zeros((n_links, 2))
    for i, (nid, rx_pos) in enumerate(sorted(rx_positions.items())):
        d_rx = np.array(rx_pos) - np.array(person_pos)
        d_rx_norm = d_rx / (np.linalg.norm(d_rx) + 1e-10)
        # Doppler = v · (d_tx_hat + d_rx_hat) / lambda
        projection = (d_tx_norm + d_rx_norm) / wavelength
        A[i] = projection

    # Rotate to body coordinates
    cos_o = np.cos(person_orientation)
    sin_o = np.sin(person_orientation)
    R = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
    A_body = A @ R.T

    # For each time step, extract dominant Doppler and solve for velocity
    T_prime = list(doppler_profiles.values())[0].shape[1]
    bvp = np.zeros((2, T_prime))

    for t in range(T_prime):
        # Extract dominant Doppler frequency per link
        dopplers = np.zeros(n_links)
        for i, (nid, dp) in enumerate(sorted(doppler_profiles.items())):
            freq_idx = np.argmax(dp[:, t])
            dopplers[i] = freq_idx  # Map to actual Hz

        # Least-squares solve: dopplers = A_body @ v
        v_body, _, _, _ = np.linalg.lstsq(A_body, dopplers, rcond=None)
        bvp[:, t] = v_body

    return bvp
```

**Important:** BVP requires knowing (or estimating) the person's position and orientation. With your 3-RX topology, position can be estimated from RSSI or CSI-based localization. Orientation is harder — Widar3.0 assumes it's known or uses a coarse estimate. For your system, start with spectrograms (no position required) and add BVP as a second-stage feature when localization is working.

---

## 3. Model Architectures

### 3.1 Architecture Selection Guide

The SenseFi benchmark and subsequent research have established clear tradeoffs:

| Architecture | In-Domain Accuracy | Cross-Domain | Latency | Best For |
|-------------|-------------------|--------------|---------|----------|
| **2D-CNN (ResNet18)** | 95-98% | 60-75% | ~5 ms | Spectrogram classification |
| **LSTM/GRU** | 90-95% | 55-70% | ~10 ms | Temporal sequences |
| **CNN-GRU hybrid** | 95-97% | 80-90% (with BVP) | ~15 ms | Cross-domain gestures |
| **Transformer** | 96-99% | 70-85% | ~20 ms | Large datasets, attention |
| **PA-CSI (attention fusion)** | 98-99% | 85-98% | ~25 ms | Multi-feature fusion |

### 3.2 Model 1: ResNet18 for CSI Spectrograms (Recommended Starting Point)

A pre-trained ResNet18 modified for multi-channel CSI spectrogram input is the best starting architecture. It's fast, well-understood, and achieves strong in-domain accuracy.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class CSIResNet(nn.Module):
    """
    ResNet18 adapted for multi-channel CSI spectrogram classification.

    Input: [batch, C, H, W] where C = n_nodes * n_pcs_per_node
    Output: [batch, n_classes] activity probabilities
    """

    def __init__(self, n_input_channels=30, n_classes=7, pretrained=True):
        """
        Args:
            n_input_channels: Number of spectrogram channels
                              (3 nodes × 10 PCs = 30)
            n_classes: Number of activity classes
            pretrained: Use ImageNet pre-trained weights (recommended)
        """
        super().__init__()

        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Replace first conv layer to accept n_input_channels
        # (ImageNet uses 3 channels; we have 30)
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        # Initialize new conv1 by averaging pretrained weights across channels
        if pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.data
                # Repeat 3-channel weights to fill n_input_channels
                repeats = n_input_channels // 3 + 1
                expanded = pretrained_weight.repeat(1, repeats, 1, 1)
                self.conv1.weight.data = expanded[:, :n_input_channels, :, :]

        # Copy remaining ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # New classification head
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Usage
model = CSIResNet(n_input_channels=30, n_classes=7).cuda()
```

### 3.3 Model 2: CNN-GRU Hybrid (Widar3.0 Architecture)

This architecture processes spatial features with CNN and temporal dynamics with GRU, matching Widar3.0's design for BVP or Doppler sequences:

```python
class CNNGRU(nn.Module):
    """
    CNN-GRU hybrid for temporal CSI sequence classification.
    Based on Widar3.0's architecture.

    Input: [batch, T, C, H, W] — sequence of T spectrogram frames
    Output: [batch, n_classes]
    """

    def __init__(self, n_input_channels=3, n_classes=7, hidden_dim=128):
        super().__init__()

        # Spatial feature extractor (per-frame CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Temporal sequence model
        self.gru = nn.GRU(
            input_size=128 * 4 * 4,  # Flattened CNN output
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, n_classes),  # *2 for bidirectional
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Apply CNN to each time step
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1)  # [B, T, 2048]

        # GRU over temporal sequence
        gru_out, _ = self.gru(features)  # [B, T, hidden*2]

        # Use last time step's output
        out = gru_out[:, -1, :]
        return self.fc(out)
```

### 3.4 Model 3: CSI Transformer (State of the Art)

Transformers have become competitive with CNN-GRU hybrids for WiFi sensing. The self-attention mechanism captures long-range temporal dependencies that GRU struggles with, and can model inter-subcarrier relationships.

```python
class CSITransformer(nn.Module):
    """
    Transformer encoder for CSI activity recognition.
    Processes the PCA-reduced time series directly.

    Input: [batch, T, D] where D = n_nodes * n_pcs
    Output: [batch, n_classes]
    """

    def __init__(self, input_dim=60, n_classes=7, d_model=128,
                 nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Project input to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        B, T, D = x.shape

        # Project and add positional encoding
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]

        # Transformer encoding
        x = self.transformer(x)

        # Use CLS token output for classification
        cls_out = x[:, 0, :]
        return self.fc(cls_out)
```

---

## 4. Multi-Node Sensor Fusion

### 4.1 Fusion Strategies

With 3 RX nodes, you have three approaches to combining their information:

**Early Fusion (recommended for simplicity):** Concatenate all nodes' features into a single input tensor before the model. This is what the spectrogram channel approach in Section 2.2 does — each node's PCA components become channels in the image.

**Late Fusion:** Train separate feature extractors per node, then combine their outputs before classification. More parameters, but each node can specialize.

**Attention-based Fusion:** Use a cross-attention mechanism to let nodes attend to each other. The most powerful approach for localization where inter-node relationships encode spatial information.

### 4.2 Attention-based Multi-Node Fusion Network

```python
class MultiNodeFusion(nn.Module):
    """
    Attention-based fusion for 3 RX nodes.
    Each node's features attend to the other nodes,
    learning spatial relationships for localization.
    """

    def __init__(self, feature_dim=256, n_nodes=3, n_classes=7, n_heads=4):
        super().__init__()
        self.n_nodes = n_nodes

        # Per-node feature extractors (shared or independent)
        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(108, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
            ) for _ in range(n_nodes)
        ])

        # Cross-node attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * n_nodes, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_features):
        """
        Args:
            node_features: list of [B, T, 108] tensors, one per node
        """
        # Encode each node
        encoded = []
        for i, feat in enumerate(node_features):
            # Average pool over time
            feat_avg = feat.mean(dim=1)  # [B, 108]
            enc = self.node_encoders[i](feat_avg)  # [B, feature_dim]
            encoded.append(enc)

        # Stack for attention: [B, n_nodes, feature_dim]
        stacked = torch.stack(encoded, dim=1)

        # Cross-node self-attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Flatten and classify
        fused = attended.reshape(attended.size(0), -1)
        return self.classifier(fused)
```

### 4.3 Localization with Multi-Node CSI

For pinpointing a person's position, the model must learn the spatial relationship between CSI changes across nodes. The simplest approach is regression: predict (x, y) coordinates from the multi-node CSI feature vector.

```python
class CSILocalizer(nn.Module):
    """
    Regress (x, y) room position from multi-node CSI.

    Uses the insight that each TX-RX link's CSI change is related
    to the person's distance from that link's Fresnel zone.
    """

    def __init__(self, n_nodes=3, n_subcarriers=108, hidden_dim=256):
        super().__init__()

        input_dim = n_nodes * n_subcarriers  # 324

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

        # Regression head for (x, y) coordinates
        self.position_head = nn.Linear(128, 2)

        # Optional: classification head for zone-based localization
        # (divide room into grid cells, classify which cell)
        self.zone_head = nn.Linear(128, 16)  # 4×4 grid

    def forward(self, x, mode='regression'):
        """
        Args:
            x: [B, n_nodes * n_subcarriers] concatenated amplitude features
            mode: 'regression' for (x,y) or 'zone' for grid classification
        """
        features = self.encoder(x)

        if mode == 'regression':
            return self.position_head(features)  # [B, 2]
        else:
            return self.zone_head(features)  # [B, 16]
```

---

## 5. Domain Adaptation — The Hardest Problem

### 5.1 Why This Is Hard

When you move your ESP32-S3 sensors to a new room, everything changes: wall positions alter multipath reflections, different furniture creates new scattering patterns, and even the same activity produces different CSI signatures. A model trained in Room A may drop from 95% to 40% accuracy in Room B without adaptation.

The domain shift has multiple axes:

| Domain Factor | What Changes | Impact |
|--------------|--------------|--------|
| **Environment** | Room geometry, furniture, walls | Changes all multipath patterns |
| **Location** | Person's position in room | Changes Fresnel zone intersection |
| **Orientation** | Person facing direction | Changes which body parts reflect |
| **Subject** | Different person's body | Changes scattering cross-section |

### 5.2 Strategy 1: Domain-Independent Features (BVP)

The Widar3.0 approach: transform CSI into features that are physically invariant to the environment. BVP captures the person's velocity in body coordinates, which is the same regardless of room geometry.

Results from the literature: Widar3.0 achieves ~89.7% accuracy across locations, ~82.6% across orientations, and ~92.4% across environments without retraining. This is the gold standard for zero-effort cross-domain generalization with 6 links. With your 3 links, accuracy will be lower but still viable — recent work shows DFS (Doppler) profiles fed directly into deep networks can achieve competitive results with fewer links by leveraging temporal context.

### 5.3 Strategy 2: Adversarial Domain Adaptation

Train a feature extractor that produces domain-invariant representations by adding a domain discriminator that tries to identify which room the data came from. The feature extractor is trained to fool the discriminator (adversarial objective) while still correctly classifying activities.

```python
class DomainAdaptiveCSINet(nn.Module):
    """
    Adversarial domain adaptation for cross-room CSI classification.

    Architecture:
    - Shared feature extractor (ResNet backbone)
    - Activity classifier (main task)
    - Domain discriminator (adversarial — tries to identify the room)

    Training uses gradient reversal: the feature extractor is trained
    to maximize domain classification loss (confuse the discriminator)
    while minimizing activity classification loss.
    """

    def __init__(self, n_input_channels=30, n_classes=7, n_domains=3):
        super().__init__()

        # Shared feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, 7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )

        # Activity classifier
        self.activity_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

        # Domain discriminator (with gradient reversal)
        self.domain_discriminator = nn.Sequential(
            GradientReversal(lambda_=1.0),  # Reverses gradients during backprop
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_domains),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        activity_pred = self.activity_classifier(features)
        domain_pred = self.domain_discriminator(features)
        return activity_pred, domain_pred


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

**Training loop for domain adaptation:**

```python
def train_domain_adaptive(model, source_loader, target_loader,
                          optimizer, n_epochs=50, device='cuda'):
    """
    Train with both source (labeled) and target (unlabeled) domain data.

    Source domain: Room A with activity labels
    Target domain: Room B with NO labels (just raw CSI)
    """
    activity_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for (src_data, src_labels), (tgt_data, _) in zip(
            source_loader, target_loader
        ):
            src_data = src_data.to(device)
            src_labels = src_labels.to(device)
            tgt_data = tgt_data.to(device)

            # Source domain: labeled data
            src_activity, src_domain = model(src_data)
            activity_loss = activity_criterion(src_activity, src_labels)
            src_domain_loss = domain_criterion(
                src_domain, torch.zeros(len(src_data), dtype=torch.long).to(device)
            )

            # Target domain: unlabeled data (only domain loss)
            _, tgt_domain = model(tgt_data)
            tgt_domain_loss = domain_criterion(
                tgt_domain, torch.ones(len(tgt_data), dtype=torch.long).to(device)
            )

            # Combined loss
            loss = activity_loss + 0.5 * (src_domain_loss + tgt_domain_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss:.4f}")
```

### 5.4 Strategy 3: Few-Shot Meta-Learning

When you deploy in a new room, collect just 5-10 labeled examples per activity (a 30-second calibration walk). Meta-learning (MAML or Prototypical Networks) can adapt the model to the new domain with this minimal data.

```python
class PrototypicalCSINet(nn.Module):
    """
    Prototypical Network for few-shot CSI activity recognition.

    During meta-training: learns an embedding space where
    same-activity CSI from different rooms clusters together.

    During deployment: collect K "support" examples per class,
    compute class prototypes, classify new data by nearest prototype.
    """

    def __init__(self, n_input_channels=30, embedding_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def classify_few_shot(self, support_set, query, n_classes):
        """
        Args:
            support_set: dict {class_label: [K, C, H, W] support examples}
            query: [B, C, H, W] query examples to classify

        Returns:
            predictions: [B] predicted class labels
            distances: [B, n_classes] distance to each prototype
        """
        # Compute prototype (mean embedding) for each class
        prototypes = {}
        for label, examples in support_set.items():
            embeddings = self.encoder(examples)
            prototypes[label] = embeddings.mean(dim=0)

        proto_stack = torch.stack([prototypes[l] for l in sorted(prototypes)])

        # Embed query
        query_emb = self.encoder(query)

        # Compute distances to prototypes
        distances = torch.cdist(query_emb, proto_stack.unsqueeze(0)).squeeze(0)

        # Classify by nearest prototype
        predictions = distances.argmin(dim=1)
        return predictions, -distances  # Negative distance as logits
```

### 5.5 Strategy 4: Contrastive Self-Supervised Pre-training

Pre-train the feature extractor on unlabeled CSI from many rooms using contrastive learning. The key insight: two spectrograms from different RX nodes at the same time instant are a "positive pair" (same activity, different view), while spectrograms from different time instants are "negative pairs."

This learns environment-invariant features without any labels, then fine-tune on a small labeled dataset.

```python
class ContrastiveCSIPretrainer(nn.Module):
    """
    SimCLR-style contrastive pre-training for CSI features.

    Positive pairs: spectrograms from different RX nodes at same time
    Negative pairs: spectrograms from different time windows
    """

    def __init__(self, encoder, projection_dim=64):
        super().__init__()
        self.encoder = encoder  # Shared backbone (e.g., ResNet18)

        # Projection head (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, C, H, W] spectrogram from node A
            x2: [B, C, H, W] spectrogram from node B (same time)
        """
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2

    def contrastive_loss(self, z1, z2, temperature=0.5):
        """NT-Xent loss (normalized temperature-scaled cross entropy)."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        z = nn.functional.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.mm(z, z.T) / temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2*B), torch.arange(0, B)
        ]).to(z.device)

        return nn.functional.cross_entropy(sim, labels)
```

---

## 6. Training Pipeline on RTX 4080

### 6.1 Dataset Collection Protocol

```
Phase 1: Collect labeled data in primary room
  - 5 activities × 60 seconds each × 3 repetitions = 15 minutes
  - Activities: empty, walking, sitting, standing, falling
  - Vary person position and orientation each repetition

Phase 2: (Optional) Collect unlabeled data in 2-3 additional rooms
  - Just walk around for 5 minutes per room
  - No labels needed (for contrastive pre-training or domain adaptation)

Phase 3: Few-shot calibration in deployment room
  - 10 seconds per activity = 50 seconds total
  - Used as support set for Prototypical Networks
```

### 6.2 Complete Training Script

```python
import torch
from torch.utils.data import DataLoader, Dataset

class CSISpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.specs = spectrograms  # [N, C, H, W]
        self.labels = labels       # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]


def train_csi_model(model, train_data, val_data, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)

    best_val_acc = 0
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: loss={train_loss/len(train_loader):.4f} "
              f"train_acc={correct/total:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

    return best_val_acc


# Example configuration
config = {
    'lr': 1e-3,
    'epochs': 100,
    'batch_size': 32,
}
```

### 6.3 Real-Time Inference Pipeline

```python
class RealtimeCSIClassifier:
    """
    Real-time inference on RTX 4080.
    Receives tensors from Pi via ZMQ, classifies, returns results.
    """

    def __init__(self, model_path, n_classes=7, device='cuda'):
        self.device = torch.device(device)
        self.model = CSIResNet(n_input_channels=30, n_classes=n_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.class_names = [
            'empty', 'walking', 'sitting', 'standing',
            'falling', 'gesture', 'breathing'
        ]

        # Warm up GPU
        dummy = torch.randn(1, 30, 65, 32).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    @torch.no_grad()
    def predict(self, spectrogram_tensor):
        """
        Args:
            spectrogram_tensor: numpy array [C, H, W] from Pi

        Returns:
            activity: string, predicted activity
            confidence: float, softmax probability
            latency_ms: float, inference time
        """
        import time
        t0 = time.perf_counter()

        x = torch.FloatTensor(spectrogram_tensor).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        latency_ms = (time.perf_counter() - t0) * 1000

        return self.class_names[pred_idx], confidence, latency_ms
```

---

## 7. Key Research Papers for Module 4

### Benchmarks & Model Comparisons

1. **SenseFi: WiFi CSI Sensing Benchmark**
   Yang, C. et al. *Patterns* 4(3), 2023.
   https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
   — Benchmarks MLP, CNN, RNN, LSTM, GRU, Transformer, CNN-RNN across 3 datasets. Open-source PyTorch library. The starting point for model selection.

2. **Deep Learning for WiFi Human Sensing**
   Yang, C. et al. arXiv:2207.07859 (2022).
   — Companion paper to SenseFi. Includes transfer learning and unsupervised learning results.

### Domain Adaptation & Cross-Domain

3. **Widar3.0: Zero-Effort Cross-Domain Gesture Recognition**
   Zheng, Y. et al. *IEEE TPAMI* 44(11), 2022.
   — Introduces BVP and the CNN-GRU architecture. Achieves 89.7% / 82.6% / 92.4% across locations, orientations, and environments.

4. **Wi-CBR: Cross-domain Behavior Recognition via Multimodal Awareness**
   arXiv:2506.11616 (2025).
   — State-of-the-art cross-domain results on Widar3.0 and XRF55 datasets. Uses phase and DFS modality fusion with cross-attention. Outperforms BVP-based methods.

5. **GesFi: Beyond Physical Labels for Domain-Robust Gesture Recognition**
   Zhang et al. *Proc. ACM IMWUT*, Aug 2025.
   — Proposes learning latent domains instead of physical domain labels. Shows that physical domain boundaries are suboptimal for adversarial generalization.

6. **PA-CSI: Phase-Amplitude Attention Network**
   *Sensors* 25(4), Feb 2025.
   — Dual-feature (amplitude + phase) fusion with Gated Residual Networks. Achieves 99.9% on StanWiFi, 98.0% on MultiEnv.

### Contrastive & Self-Supervised Learning

7. **Self-Supervised WiFi-Based Activity Recognition**
   Brinke et al. arXiv:2104.09072 (2021).
   — Uses multi-view contrastive learning (different RX nodes = different views). AlexNet, VGG16, ResNet18 encoders benchmarked.

8. **Cross-Domain Federated Feature Fusion**
   *Arabian J. Sci. Eng.* (2025).
   — Federated learning + uncertainty-aware feature fusion. Achieves 96.5% cross-domain accuracy on Widar3.0.

### Transformer Architectures

9. **WiTransformer: Robust Gesture Recognition**
   *Sensors* 23(5), 2023.
   — Transformer backbone for WiFi gesture recognition. Addresses GRU limitations in capturing long-range dependencies. Competitive with Widar3.0 CNN-GRU.

### Datasets

10. **Widar3.0 Dataset** — https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
    258K gesture instances, 75 domains, includes raw CSI + DFS + BVP.

11. **MM-Fi Dataset** — Multi-modal (CSI + RGB-D + LiDAR + mmWave), 40 subjects, 4 scenarios.

12. **UT-HAR Dataset** — University of Texas HAR dataset, Intel 5300 NIC, 30 subcarriers.

---

## 8. Recommended Implementation Roadmap

**Phase 1 (Week 1-2): Baseline**
- Collect labeled data in one room (5 activities, 3 repetitions)
- Train ResNet18 on CSI spectrograms
- Target: >90% in-domain accuracy

**Phase 2 (Week 3-4): Multi-Node Fusion**
- Implement early fusion (multi-channel spectrograms)
- Add localization head (zone classification)
- Benchmark against single-node performance

**Phase 3 (Week 5-6): Domain Adaptation**
- Collect unlabeled data in 2 additional rooms
- Implement adversarial domain adaptation
- Implement few-shot calibration with Prototypical Networks
- Target: >80% cross-room accuracy with 10-shot calibration

**Phase 4 (Week 7-8): Advanced Features**
- Implement BVP extraction (requires localization from Phase 2)
- Train CNN-GRU on BVP sequences
- Contrastive pre-training with multi-node views
- Target: >85% zero-effort cross-room accuracy

---

*Module 4 Complete — Full pipeline from RF physics to deployed deep learning system.*
