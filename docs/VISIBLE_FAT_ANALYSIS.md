# Visible Fat & Marbling Analysis (Module 3)

**Project:** Marbl — AI Meat Quality Control & Predictive Shelf-Life Inspection Pipeline  
**Document:** Surface Lipid Quantification & Semantic Marbling Segmentation Specification  

---

## 1. Current Operational Pipeline: Colorimetric Lipid Quantification

Surface fat coverage in the active Marbl production pipeline is computed through an unsupervised foreground-masked colorimetric algorithm in [`backend/fat_analyzer.py`](backend/fat_analyzer.py).

### 1.1 Algorithmic Workflow

```mermaid
flowchart LR
    A["Raw RGB Image"] --> B["GrabCut Foreground Mask"]
    B --> C["Extract Meat ROI<br/>(Exclude Background)"]
    C --> D["Color Space Conversion:<br/>RGB ──► HSV & CIE-LAB"]
    D --> E["Dual-Condition Masking:<br/>Lightness L* > 165 & Saturation S < 0.25"]
    E --> F["Morphological Cleanup:<br/>3x3 Kernel Opening & Closing"]
    F --> G["Compute Visible Fat %<br/>(Fat Pixels / Total Meat Pixels)"]
    G --> H["Render Dual View:<br/>Original vs. AI Segmented Map"]
```

### 1.2 Mathematical Formulation & Color Thresholds
1. **Foreground Extraction:** The binary GrabCut mask eliminates trays, conveyor belts, cutting boards, and fiducial markers:
   $$\mathcal{M}_{\text{foreground}}(x, y) \in \{0, 1\}$$
2. **Color Space Transformations:** RGB pixels are converted to HSV (Hue-Saturation-Value) and CIE-LAB:
   $$(H, S, V) = \mathcal{T}_{\text{HSV}}(R, G, B), \quad (L^*, a^*, b^*) = \mathcal{T}_{\text{LAB}}(R, G, B)$$
3. **Dual-Condition Lipid Segmentation:**
   - **Lipid Tissue (Subcutaneous & Intramuscular Fat):** Characterized by high lightness, low saturation, and high value:
     $$\mathcal{M}_{\text{fat}}(x, y) = \mathcal{M}_{\text{foreground}}(x, y) \land (S < 0.25) \land (L^* > 165) \land (V > 0.65)$$
   - **Lean Muscle Tissue:** Characterized by strong red chrominance and moderate-to-high saturation:
     $$\mathcal{M}_{\text{lean}}(x, y) = \mathcal{M}_{\text{foreground}}(x, y) \land \neg \mathcal{M}_{\text{fat}}(x, y) \land (a^* > 135)$$
4. **Morphological Post-Processing:**
   - A $3 \times 3$ elliptical structuring element performs morphological **opening** ($\mathcal{M}_{\text{fat}} \circ K$) to eliminate isolated white specular reflections caused by moisture on the meat surface.
   - Morphological **closing** ($\mathcal{M}_{\text{fat}} \bullet K$) bridges microscopic gaps along contiguous lipid strands.
5. **Visible Fat Percentage Metric:**
   $$\text{Visible Fat Coverage (\%)} = \frac{\sum_{x, y} \mathcal{M}_{\text{fat}}(x, y)}{\sum_{x, y} \mathcal{M}_{\text{foreground}}(x, y)} \times 100\%$$
   $$\text{Lean Meat Coverage (\%)} = 100\% - \text{Visible Fat Coverage (\%関)}$$

---

## 2. Proposed Deep Semantic Fat Segmentation Pipeline

While colorimetry is real-time and requires zero manual training labels, it cannot differentiate between superficial subcutaneous fat caps, heavy seam fat, and fine intramuscular marbling flecks. The proposed deep learning segmentation pipeline addresses this.

### 2.1 Model Architecture: MobileNetV3-UNet
To maintain real-time performance on edge devices, the architecture employs a **MobileNetV3-Large** encoder paired with a lightweight **U-Net** decoder equipped with depthwise separable transpose convolutions:

```
Encoder (MobileNetV3-Large Backbone)
Level 1 (stride 2)  ──► [Skip 1] ──┐
Level 2 (stride 4)  ──► [Skip 2] ─┐│
Level 3 (stride 8)  ──► [Skip 3] ┐││
Level 4 (stride 16) ──► [Skip 4] │││
Level 5 (stride 32) ───────────┐ │││
                               ▼ ▼▼▼
Decoder: Bilinear Upsample + Concat + Depthwise Separable Conv
                               │
                               ▼
Final 1x1 Conv ──► 4-Class Softmax Probability Map (H x W x 4)
```

### 2.2 Target Class Taxonomy

| Class ID | Class Name | Definition | Visual Appearance |
|---|---|---|---|
| **0** | `background` | Trays, conveyors, cutting boards, voids | Non-meat areas |
| **1** | `lean_muscle` | Red and pink contractile muscle tissue | Deep crimson, pink, burgundy |
| **2** | `seam_subcutaneous_fat` | Heavy external fat cap and intermuscular deposits | Solid white, ivory, cream |
| **3** | `intramuscular_marbling`| Microscopic lipid flecks dispersed within muscle fibers | Delicate white spider-web striations |

### 2.3 Dataset Preparation & Annotation Methodology
- **Dataset Size:** 600 high-resolution meat cross-sections (beef ribeye, striploin, and pork loin).
- **Annotation Tool:** VGG Image Annotator (VIA) and LabelMe with polygon boundary traces. Detailed instructions are documented in [`docs/fat_annotation_guide.md`](fat_annotation_guide.md).
- **Split Strategy:** 420 images (70%) train, 90 images (15%) validation, 90 images (15%) test.

### 2.4 Loss Function Formulation
To address class imbalance between large muscle areas and fine marbling flecks, training uses a composite loss combining **Class-Weighted Cross-Entropy** and **Soft Dice Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{WCE}} + \lambda \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{WCE}} = -\sum_{c=1}^4 w_c \cdot y_c \log(\hat{y}_c), \quad \text{where } w_{\text{marbling}} = 3.5, \, w_{\text{fat}} = 1.8, \, w_{\text{lean}} = 1.0$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

### 2.5 Quantitative Evaluation Criteria

| Evaluation Criterion | Target Specification | Validation Protocol |
|---|---|---|
| **Mean Intersection-over-Union (mIoU)** | $\ge 0.85$ across all classes | 90-image test set evaluation |
| **Marbling Class IoU** | $\ge 0.78$ | Micro-fleck IoU evaluation |
| **Inference Latency** | $\le 45\text{ ms}$ on NVIDIA GPU | TensorRT FP16 export |
| **Chemical Correlation ($R^2$)** | $R^2 \ge 0.92$ | Comparison with AOAC 991.36 Soxhlet Ether Extraction |

