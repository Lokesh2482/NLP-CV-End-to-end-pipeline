# Pokemon Tactical Strike - AI Targeting System

A multi-modal AI pipeline combining Natural Language Processing and Computer Vision for intelligent target classification and localization in battlefield scenarios.

## Problem Overview

This system processes tactical orders from headquarters and analyzes battlefield imagery to:
- Parse mission directives from complex textual orders
- Identify Pokemon species and their locations in images
- Generate precise targeting coordinates based on mission parameters
- Minimize collateral damage by distinguishing targets from protected species

## System Architecture

### 1. Natural Language Processing Module (`NeoBert_Test.ipynb`)

**Purpose**: Extract target classification from tactical mission briefings

**Approach**:
- Fine-tuned NeoBERT transformer model for sequence classification
- Handles diverse communication styles and military jargon
- Classifies target Pokemon species from complex, multi-sentence orders

**Key Features**:
- Comprehensive synonym mapping for robust species recognition
- Synthetic data generation with tactical filler phrases
- Training on 10,000+ diverse prompt variations
- Handles negations, distractors, and ambiguous instructions

**Model Configuration**:
- Base Model: `chandar-lab/NeoBERT`
- Task: 4-class classification (Pikachu, Charizard, Bulbasaur, Mewtwo)
- Optimizer: AdamW with weight decay (0.01) for regularization
- Training: 2 epochs with early stopping
- Learning Rate: 2e-5

### 2. Computer Vision Module (`best_scoring_model.ipynb`)

**Purpose**: Detect and localize Pokemon species in battlefield imagery

**Approach**:
- RF-DETR (Real-time DEtection TRansformer) Nano architecture
- Custom-trained on Pokemon detection dataset
- Fine-tuned for 4 Pokemon species classification

**Key Features**:
- Handles varying scales, orientations, and partial occlusions
- Robust bounding box prediction with confidence scoring
- Detection threshold: 0.2 for optimal recall
- Center-of-mass calculation for precise targeting coordinates
- Color-coded visualization per species

**Model Configuration**:
- Architecture: RF-DETR Nano
- Training: 10 epochs, batch size 4
- Gradient accumulation steps: 4
- Checkpoint: `output/checkpoint_best_total.pth`

**Class Mapping**:
```python
CLASS_NAME_TO_ID = {
    "bulbasaur": 1,
    "charizard": 2,
    "mewtwo": 3,
    "pikachu": 4
}
```

### 3. Decision Fusion Engine

**Pipeline Flow**:
1. NLP module identifies target species from tactical orders
2. CV module detects all Pokemon in battlefield image
3. Fusion engine filters detections to match target species only
4. Targeting coordinates generated from filtered bounding boxes
5. Output formatted as JSON array of [x, y] coordinates

## Installation

### Requirements

```bash
# Core dependencies
pip install transformers==4.38.1 torch xformers==0.0.28.post3 flash-attn
pip install rfdetr==1.2.1 supervision==0.26.1 roboflow
pip install numpy scikit-learn tqdm pillow

# For notebook environment
pip install jupyter ipywidgets
```

### Setup

1. Clone the repository
2. Install dependencies
3. Configure Roboflow API key:
   ```python
   from google.colab import userdata
   os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
   ```

## Usage

### Running the Pipeline

#### 1. NLP Module (`NeoBert_Test.ipynb`)

Open and run the notebook to:
- Load and process training prompts from `train_prompts.json`
- Generate 10,000 synthetic training examples
- Train NeoBERT classifier
- Process test prompts and extract target species
- Generate `simplified_test_prompts.json` with target classifications

**Output Format**:
```json
{
  "images": [
    {
      "id": "img_00000",
      "target": ["pikachu"],
      "protected": ["charizard", "bulbasaur", "mewtwo"]
    }
  ]
}
```

#### 2. CV Module (`best_scoring_model.ipynb`)

Open and run the notebook to:
- Download and prepare COCO-format dataset
- Train RF-DETR model (or load pretrained weights)
- Run inference on test images
- Generate targeting coordinates
- Create visualizations with bounding boxes
- Output `test_predictions.csv`

**Final Output Format**:
```csv
image_id,points
img_00000.png,"[[445.05, 369.73], [286.51, 402.06]]"
img_00001.png,"[[371.79, 140.69], [48.61, 210.21]]"
```

### Key Functions

**NLP Module**:
```python
# Generate synthetic training data
processor = PromptProcessor()
synthetic_data = processor.generate_synthetic_prompts(num_prompts=10000)

# Predict targets from prompts
predicted_targets = predict_targets(model, tokenizer, test_prompts, device='cuda')
```

**CV Module**:
```python
# Load trained model
model = RFDETRNano(num_classes=4, pretrain_weights="output/checkpoint_best_total.pth")

# Run detection
detections = model.predict(image, threshold=0.2)

# Filter by target class
target_bboxes = filter_detections_by_class(detections, target_class_id)

# Compute targeting coordinates
targeting_points = [compute_center_of_mass(bbox) for bbox in target_bboxes]
```

## Dataset Structure

```
train_data/
├── annotations/
│   ├── instances_train.json    # Pokemon location annotations (COCO format)
│   └── train_prompts.json      # Sample tactical orders
└── images/                     # Training imagery

test_data/
├── test_prompts.json           # Test tactical orders
└── [test images]               # Test battlefield imagery

output/
├── checkpoint_best_total.pth   # Best CV model checkpoint
└── simplified_test_prompts.json # NLP predictions

test_visualization/             # Annotated detection results
test_predictions.csv            # Final submission file
```

## Evaluation Metrics

**Scoring System**:
- **+1 point**: Direct hit on designated target species
- **+1 point**: Eliminating all enemy Pokemon in image
- **-1 point**: Collateral damage (hit on protected species)
- **-1 point**: Ammunition waste (every 3 missed shots)

**Final Score**: `(Correct Hits) - (Incorrect Hits) - (Misses / 3)`

**Assessment Rubric**:
- 70% Accuracy (targeting precision and mission order interpretation)
- 20% Technical Innovation (novel approaches and architectures)
- 10% Code Quality (documentation and modularity)

## Technical Highlights

### NLP Innovations
- **10 diverse prompt generation strategies**: tactical reports, buried instructions, negations, distractors, ambiguous orders
- **Comprehensive synonym database**: 15+ synonyms per Pokemon species
- **Tactical filler phrases**: 50+ realistic military communication patterns
- **Robust training**: Balanced dataset with stratified train/val split
- **Early stopping**: Prevents overfitting on synthetic data

### CV Innovations
- **State-of-the-art detector**: RF-DETR transformer architecture
- **Class-specific visualization**: Color-coded bounding boxes (Yellow: Pikachu, Orange-Red: Charizard, Green: Bulbasaur, Purple: Mewtwo)
- **Adaptive thresholding**: Confidence-based detection filtering
- **Optimal text scaling**: Automatic annotation parameters based on image resolution
- **Memory management**: GPU cleanup utilities for efficient resource usage

### Integration Strategy
- Clean separation between NLP and CV modules
- JSON-based inter-module communication
- Efficient batch processing for test sets
- Comprehensive error handling and logging

## Repository Structure

```
pokemon-tactical-strike/
├── NeoBert_Test.ipynb              # NLP training and inference
├── best_scoring_model.ipynb        # CV training and inference
├── pokemon_nlp_model/              # Saved NLP model weights
├── output/                         # CV model checkpoints
│   └── checkpoint_best_total.pth
├── test_visualization/             # Annotated output images
├── simplified_test_prompts.json    # NLP module output
└── test_predictions.csv            # Final submission file
```

## Performance Statistics

**NLP Module**:
- Training samples: 10,000+ (synthetic + original)
- Validation accuracy: Reported in notebook output
- Processing speed: ~100 prompts/second on GPU

**CV Module**:
- Training epochs: 10
- Detection threshold: 0.2
- Successfully detects targets across varying orientations and scales
- Per-class detection statistics logged in notebook

## Limitations and Future Work

- **Detection threshold**: May require tuning per species for optimal balance
- **Temporal context**: Current system processes images independently
- **Species coverage**: Limited to 4 Pokemon species
- **Prompt complexity**: Very long orders (1000+ words) may lose context
- **Occlusion handling**: Partial occlusions reduce detection confidence

**Potential Improvements**:
- Multi-scale feature fusion for small object detection
- Attention mechanisms for long-context NLP
- Active learning on misclassified examples
- Ensemble methods combining multiple detectors

## Technical Dependencies

**Core Libraries**:
- PyTorch (GPU support required)
- Transformers (Hugging Face)
- RF-DETR 1.2.1
- Supervision 0.26.1
- Roboflow

**Hardware Requirements**:
- GPU with 8GB+ VRAM (NVIDIA recommended)
- 16GB+ system RAM
- 10GB+ storage for datasets and models

This project demonstrates practical applications of:
- Multi-modal learning (text + vision)
- Transfer learning with pre-trained models
- Real-time object detection
- Natural language understanding

---

**Disclaimer**: The tactical/military framing is purely fictional and designed for the competition scenario. This is an educational AI project focused on multi-modal learning techniques.
