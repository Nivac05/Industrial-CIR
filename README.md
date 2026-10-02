# IC-CIR: Constraint-Aware Composed Image Retrieval for Industrial Parts

IC-CIR is a **Composed Image Retrieval (CIR)** system designed for industrial replacement-part search.

Instead of searching with only an image or only text, IC-CIR takes:

- a **reference image** of an industrial component, and
- a **natural-language modification**, such as  
  `same flange, but larger and brass`

and retrieves parts that preserve the required structure while satisfying the requested changes.

The project explores explicit **KEEP / MODIFY / NUMERIC constraints**, part-aware visual representations, disentangled attribute spaces, and constraint-aware reranking on top of OpenCLIP.

---

## Problem

Industrial part retrieval is more difficult than conventional image similarity search.

A visually similar component may still be unsuitable because of differences in:

- material
- size
- geometry
- component type
- structural details

Likewise, text-only retrieval loses the detailed geometry available in a reference image.

IC-CIR approaches the problem as:

```text
Reference Image + Modification Text
                ↓
       Composed Query Representation
                ↓
        Industrial Part Gallery
                ↓
           Top-K Matches
```

For example:

```text
Reference: metallic flange

Query:
"same flange, but larger and brass"

Expected retrieval:
a structurally similar flange satisfying the requested
material and size modification
```

---

# Architecture

IC-CIR uses a frozen **OpenCLIP ViT-B/32** backbone and trains additional modules specifically for composed industrial-part retrieval.

```text
                         ┌─────────────────────┐
Reference Image ────────►│ OpenCLIP ViT-B/32  │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
              Global Feature               Patch Tokens
                     │                             │
                     │                    Part/Structure
                     │                    Attention Branch
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                              Disentangler
                                    │
                         Structure Representation
                                    │
                                    │
Modification Text ──► OpenCLIP Text Encoder
                                    │
                         Constraint Parser
                       KEEP / MODIFY / NUMERIC
                                    │
                              Disentangler
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
               Structure         Material           Size
                   │                │                │
                   └────── Prototype Composition ───┘
                                    │
                              KEEP Gate
                                    │
                         Feature Fusion Network
                                    │
                         Composed Query Embedding
                                    │
                              Cosine Search
                                    │
                         Top-K Candidate Parts
                                    │
                     Constraint-Aware Reranking
                                    │
                             Final Results
```

---

# Core Components

## 1. OpenCLIP Feature Extraction

The system uses:

```text
ViT-B/32
Pretrained: OpenAI
Embedding dimension: 512
```

The OpenCLIP backbone remains **frozen during IC-CIR training**.

Two forms of visual information are extracted:

- global image representation
- ViT patch-level representations

A sanity check in the notebook verifies that the custom global extraction path matches OpenCLIP's official image encoder with a cosine similarity of **1.0** on the tested samples.

---

## 2. Part / Structure Branch

Industrial components are often differentiated by relatively small structural details.

Instead of relying only on CLIP's global representation, IC-CIR applies **multi-head attention over ViT patch tokens**.

A learned query attends to the patch representations and produces a dedicated part/structure-aware vector.

```text
Patch Tokens
     ↓
Multi-Head Attention
     ↓
Part / Structure Representation
```

This representation is combined with the global visual feature before attribute disentanglement.

---

## 3. Constraint Parser

Modification queries are represented using three types of constraints:

### KEEP

Attributes that should remain consistent with the reference.

Example:

```text
keep the flange type
```

### MODIFY

Categorical properties that should change.

Example:

```text
make it brass
```

### NUMERIC

Relative size modifications.

Examples:

```text
15% larger
30% larger
15% smaller
```

The experimental dataset uses discrete scale choices:

```python
[0.70, 0.85, 1.00, 1.15, 1.30]
```

The current training pipeline uses structured metadata derived from synthetically generated modification templates. Therefore, this should be considered a **controlled constraint parser**, rather than a general-purpose free-form language parser.

---

## 4. Attribute Disentanglement

IC-CIR projects features into separate learned subspaces:

```text
Structure
Material
Size
```

Each attribute is processed by its own MLP head.

The goal is to prevent a requested material change, for example, from unnecessarily destroying information about the reference component's structure.

---

## 5. Prototype Composition

Each attribute branch uses a learned prototype dictionary.

For each feature, attention is computed over **16 trainable prototypes**, producing an attribute-specific composed representation.

Separate prototype composers are used for:

- structure
- material
- size

---

## 6. KEEP-Gated Composition

The structure of the reference image and the requested structural representation are combined using a learned sigmoid gate.

Conceptually:

```text
kept_structure =
    gate × reference_structure
    +
    (1 - gate) × composed_structure
```

This allows the model to retain useful visual information from the reference while incorporating the requested modification.

---

## 7. Constraint-Aware Residual Reranking

The composed embedding first retrieves candidates using cosine similarity.

A lightweight residual reranker then adjusts the Top-K results according to explicit material constraints.

The final experimental configuration uses:

```text
Top-K candidates : 50
Material bonus   : 0.05
Size weight      : 0.00
```

Importantly, the grid search found no improvement from the tested explicit size-penalty weights. Therefore the **final deployed residual reranker uses material adjustment only**, while size information is still represented inside the learned composed-query model.

---

# Training Objectives

The full IC-CIR model combines four objectives.

### Retrieval Loss

InfoNCE aligns the composed query representation with its target image.

```text
L_retrieval = InfoNCE(query, target)
```

### Preservation Loss

Encourages structural information that should remain unchanged to stay aligned with the target.

### Modification Loss

Supervises requested attributes using:

```text
Material → Cross-Entropy Loss
Size     → Mean Squared Error
```

### Counterfactual Loss

Reference images are shuffled to construct incorrect image-text combinations.

A margin-based objective encourages the correct composed query to score above these counterfactual combinations.

The overall implementation is approximately:

```text
L =
    L_retrieval
    + 0.5 L_preservation
    + 0.5 L_modification
    + 0.3 L_counterfactual
```

---

# Dataset

## MCB_B

The main experiments use **MCB_B CAD meshes**.

The notebook identifies **25 industrial component classes**, including:

```text
bearing
bushing
castors_and_wheels
clamp
disc
fitting
flange
fork_joint
gear
handles
hinge
hook
motor
nut
pin
plate
pulley
ring
rivet
rotor
screws_and_bolts
spring
stud
switch
washer
```

CAD meshes are rendered into images using **Trimesh + Pyrender**.

---

## Synthetic Material Variants

Meshes are rendered with controlled material appearances:

```text
metallic
plastic
aluminum
brass
painted
rusted
```

Some materials use real texture images, while others are procedurally generated.

Random camera azimuths introduce viewpoint variation between rendered examples.

---

## Synthetic Size Variants

The following scale factors are used:

```text
0.70
0.85
1.00
1.15
1.30
```

These allow the model to learn queries such as:

```text
"same bearing, but smaller and metallic"

"find a similar flange that is larger and brass"

"retrieve another spring with a larger size and painted finish"
```

---

# Dataset Construction

The experiment generates:

| Split | Samples per class | Classes | Total |
|---|---:|---:|---:|
| Training | 60 | 25 | 1,500 |
| Test | 10 | 25 | 250 |

Each triplet contains:

```text
Reference image
Modification text
Target image
Material metadata
Scale metadata
Part class
```

The evaluation gallery contains the **250 rendered test targets**.

This is therefore a controlled synthetic/proxy evaluation and should not be interpreted as performance on a large-scale real-world industrial catalogue.

---

# Sample Rendered Images

<img width="577" height="860" alt="image" src="https://github.com/user-attachments/assets/cc6298e9-c189-4032-a193-7b7101958313" />


Suggested layout:

```text
Metallic | Plastic | Aluminum | Brass | Painted | Rusted
```

You can replace the placeholder below after uploading the image to the repository:

```markdown
![Rendered industrial part samples](assets/rendered_samples.png)
```

---

# Evaluation

Retrieval is evaluated using:

- **Recall@1**
- **Recall@5**
- **Recall@10**
- Median Rank
- Mean Rank

For the selected final residual-reranking configuration, the notebook reports:

| Metric | Result |
|---|---:|
| Recall@1 | **48.8%** |
| Recall@5 | **85.2%** |
| Recall@10 | **94.0%** |
| Median rank when found in Top-50 | **1.5** |
| Mean rank when found in Top-50 | **2.76** |
| Outside Top-50 | **2.4%** |

These results are measured on the **250-query synthetic MCB_B test gallery** described above.

---

## Baseline Context

The notebook also evaluates several experimental baselines and ablations.

Among the recorded results:

| Method | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Image only | 29.2% | 48.0% | 57.6% |
| Text only | 2.8% | 12.4% | 20.4% |
| Weighted image-text fusion | 26.0% | 49.6% | 59.2% |
| Simple learned Combiner | **53.6%** | 81.6% | 90.8% |
| IC-CIR, before residual reranking | 44.8% | 78.8% | 88.0% |
| **IC-CIR + residual constraint reranking** | 48.8% | **85.2%** | **94.0%** |

An important observation is that the simple learned Combiner achieves the highest Recall@1 in this controlled experiment, while IC-CIR with residual constraint reranking achieves the strongest recorded Recall@5 and Recall@10.

This distinction is intentional: the README does **not** claim that IC-CIR dominates every baseline on every metric.


```text
baseline retrieval
IC-CIR retrieval
final residual-reranked IC-CIR
```


# Qualitative Retrieval Results

The notebook found **109 modified test examples** where the ground-truth target was retrieved at **Rank 1** and the reference/target pair passed the qualitative visual-difference filter.

Example queries include:

```text
"keep the bushing type, but make it larger with painted material"

"retrieve another spring with a larger size and painted finish"

"same flange, but larger and brass"

"same hinge, but larger and plastic"
```

---

# Sample Output

<img width="1688" height="445" alt="image" src="https://github.com/user-attachments/assets/7231711f-1d04-4d36-8cf5-1429917add6f" />
<img width="1672" height="450" alt="image" src="https://github.com/user-attachments/assets/e1997b0c-01ca-4948-aef9-90d29716b7e4" />
<img width="1672" height="450" alt="image" src="https://github.com/user-attachments/assets/edb6ee55-f5bb-45de-9e3b-0199cc557ece" />
<img width="1672" height="432" alt="image" src="https://github.com/user-attachments/assets/10be4ddc-25b8-4492-a4ca-91ac6650e665" />


```text
REFERENCE
    +
GROUND-TRUTH TARGET
    +
TOP-5 RETRIEVED RESULTS
```

with the natural-language modification displayed above them.

Add it to the repository as, for example:

```text
assets/retrieval_example.png
```

Then uncomment/use:

```markdown
![IC-CIR retrieval example](assets/retrieval_example.png)
```

### Example Output Format

```text
Query:
"same flange, but larger and brass"

Reference:
Metallic flange

Requested material:
metallic → brass

Requested scale:
1.00 → 1.30

Ground-truth rank:
#1

Retrieved:
#1  Ground-truth target ✓
#2  Candidate
#3  Candidate
#4  Candidate
#5  Candidate
```

---

# Real-Image Inference

The exported model includes an inference pipeline for testing arbitrary uploaded images.

For an input image, IC-CIR evaluates:

```text
1. Original image
2. Background-removed image
3. Background-removed + cropped image
```

The strongest preprocessing variant is selected before displaying the Top-5 retrieved gallery items.

Example:

```python
run_iccir("same part but plastic")
```

or:

```python
run_iccir("same part but metallic")
```

or:

```python
run_iccir("same part but aluminum and 15% larger")
```

The inference pipeline uses `rembg` for background removal.

---

# Saved Model Structure

The notebook exports:

```text
iccir_final/
│
├── iccir_final_model.pt
├── iccir_gallery.pt
├── iccir_config.json
│
└── gallery_images/
    ├── gallery_0000.png
    ├── gallery_0001.png
    ├── ...
    └── gallery_0249.png
```

### `iccir_final_model.pt`

Contains:

- trained IC-CIR weights
- embedding dimension
- material vocabulary
- size-scale choices
- reranker configuration
- final evaluation metrics

### `iccir_gallery.pt`

Contains the gallery embeddings and associated metadata.

### `iccir_config.json`

Stores the OpenCLIP configuration and inference parameters.

---

# Tech Stack

```text
Python
PyTorch
OpenCLIP
Trimesh
Pyrender
FAISS
NumPy
Pandas
scikit-learn
Pillow
Matplotlib
rembg
ONNX Runtime
Hugging Face Datasets
```

---

# Current Limitations

This repository is an experimental proof of concept rather than a production industrial-search benchmark.

The main limitations are:

1. **Synthetic training modifications**  
   Material and size variations are generated from CAD renders rather than collected as naturally occurring replacement-part pairs.

2. **Controlled language templates**  
   Training modification text follows generated templates. The constraint parser is therefore not yet a fully learned free-form language parser.

3. **Small evaluation gallery**  
   Reported metrics use a 250-image test gallery.

4. **Synthetic material appearance**  
   Material textures approximate metallic, plastic, aluminum, brass, painted, and rusted surfaces but do not capture the full variability of real manufacturing environments.

5. **Explicit size reranking is currently inactive**  
   The grid search produced the same best retrieval scores across the tested size weights, and the selected exported configuration uses `size_weight = 0.0`.

6. **Domain gap**  
   Real photographs can contain backgrounds, lighting variation, occlusion, wear, perspective distortion, and objects outside the MCB_B classes.

The inference pipeline partially addresses the domain gap using automatic background removal and cropping.

---

# Future Work

Future extensions include:

- training on real reference/modification/target triplets
- learned free-form KEEP/MODIFY/NUMERIC parsing
- larger industrial galleries
- FAISS-based large-scale retrieval
- richer dimensional and tolerance constraints
- manufacturer and specification metadata
- improved material recognition
- multi-view CAD representations
- real-world industrial benchmark evaluation
- learned constraint-aware reranking
- stronger domain adaptation between CAD renders and photographs

---

# Project Goal

IC-CIR investigates a practical question:

> **Can industrial search preserve the identity and structure of a reference component while modifying only the attributes explicitly requested by the user?**

Rather than treating replacement-part search as generic visual similarity, the project explores retrieval where **what must remain unchanged is as important as what must change**.
