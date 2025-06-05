# Multimodal Movie Revenue Prediction System

## Overview

This system combines **text analysis** (movie plot/synopsis) and **video analysis** (YouTube trailer) using deep learning to predict movie revenue categories. The model integrates multiple modalities to provide more accurate predictions than single-modality approaches.

## Revenue Classification Categories

The model predicts revenue into 8 categories based on your existing notebook analysis:

1. **Disaster (0)** - Lowest revenue category
2. **Flop (1)** - Poor performance
3. **Successful (2)** - Moderate success
4. **Average (3)** - Average performance
5. **Hit (4)** - Good performance
6. **Outstanding (5)** - Very good performance
7. **Superhit (6)** - Exceptional performance
8. **Blockbuster (7)** - Highest revenue category

## System Architecture

### 1. Text Encoder (BERT-based)
- **Model**: BERT-base-uncased
- **Input**: Movie plot/synopsis from the `Description` field
- **Features**: 768-dimensional embeddings
- **Processing**: Tokenization → BERT encoding → Dropout → FC layer

### 2. Video Encoder (CNN-based)
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Input**: 30 frames extracted from YouTube trailers
- **Features**: 2048-dimensional embeddings
- **Processing**: Frame extraction → CNN feature extraction → Temporal pooling

### 3. Audio Encoder (1D CNN)
- **Input**: 30 seconds of audio from YouTube trailers
- **Features**: 1024-dimensional embeddings
- **Processing**: Audio extraction → 1D CNN → Global pooling

### 4. Multimodal Fusion
- **Method**: Feature concatenation + Feed-forward layers
- **Input**: Text (768) + Video (2048) + Audio (1024) = 3840 dimensions
- **Output**: 8 classes (revenue categories)
- **Architecture**: 
  - Linear(3840 → 512) → ReLU → Dropout
  - Linear(512 → 256) → ReLU → Dropout
  - Linear(256 → 8) → Softmax

## Data Splits

The system supports three different train/validation/test splits:

1. **Option 1**: 70% / 20% / 10%
2. **Option 2**: 75% / 15% / 10% 
3. **Option 3**: 80% / 10% / 10%

## Key Features

### Deep Learning Models for Video Analysis
- **Frame-level CNN**: ResNet50 for visual feature extraction
- **Temporal Processing**: Convolutional layers for sequence modeling
- **Audio CNN**: 1D convolutions for audio pattern recognition
- **Advanced Preprocessing**: Automatic frame sampling and audio normalization

### Comprehensive Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Weighted and macro averages
- **Precision & Recall**: Both weighted and macro averages
- **Confusion Matrix**: Detailed class-wise performance
- **Classification Report**: Per-class metrics

### Training Features
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Class Balancing**: Handles imbalanced dataset
- **Cross-validation**: Multiple data split testing

## Implementation Files

### 1. `multimodal_movie_revenue_predictor.py`
**Complete implementation** with:
- Full YouTube video processing
- Real-time video downloading
- Actual frame and audio extraction
- Production-ready training pipeline

### 2. `simplified_trainer.py`
**Simplified version** with:
- Simulated video/audio features for testing
- Faster development and debugging
- Reduced computational requirements
- Focus on model architecture

### 3. `requirements.txt`
Complete dependency list including:
- Deep learning frameworks (PyTorch, Transformers)
- Computer vision libraries (OpenCV, Pillow)
- Audio processing (librosa, soundfile)
- YouTube processing (pytube)
- Standard ML libraries (scikit-learn, pandas)

## Usage Instructions

### Installation
```bash
pip install -r requirements.txt
```

### Training (Full System)
```bash
python multimodal_movie_revenue_predictor.py
```

### Training (Simplified)
```bash
python simplified_trainer.py
```

## Model Performance

The system outputs comprehensive metrics for each data split:

### Training Metrics
- Loss curves for train/validation
- Accuracy progression
- Learning rate scheduling visualization

### Test Evaluation
- Final accuracy scores
- F1 scores (weighted and macro)
- Precision and recall metrics
- Detailed confusion matrices
- Per-class classification reports

### Comparison Analysis
- Side-by-side comparison of all three data splits
- Best performing split identification
- Metric trend analysis

## Technical Specifications

### Model Parameters
- **Text Encoder**: ~110M parameters (BERT-base)
- **Video Encoder**: ~25M parameters (ResNet50)
- **Audio Encoder**: ~2M parameters (1D CNN)
- **Fusion Layers**: ~2M parameters
- **Total**: ~140M parameters

### Training Configuration
- **Batch Size**: 16 (adjustable)
- **Learning Rate**: 1e-4 with scheduling
- **Optimizer**: AdamW with weight decay
- **Max Epochs**: 50 with early stopping
- **Patience**: 10 epochs

### Hardware Requirements
- **GPU**: Recommended for training (CUDA support)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for video downloads and models

## Output Files

After training, the system generates:

1. **Model Weights**: `best_multimodal_model.pth`
2. **Training Visualization**: `training_curves.png`
3. **Evaluation Results**: `evaluation_results.json`
4. **Split Comparison**: `split_comparison.csv`
5. **Confusion Matrix**: `confusion_matrix.png`

## Advanced Features

### Video Processing Pipeline
1. **URL Parsing**: Extract YouTube video IDs
2. **Video Download**: Automatic trailer downloading
3. **Frame Extraction**: Intelligent frame sampling
4. **Audio Extraction**: High-quality audio processing
5. **Feature Normalization**: Consistent preprocessing

### Text Processing Pipeline
1. **BERT Tokenization**: Advanced text encoding
2. **Attention Mechanisms**: Context-aware processing
3. **Fine-tuning**: Domain-specific adaptation
4. **Regularization**: Dropout and weight decay

### Evaluation Pipeline
1. **Stratified Splitting**: Balanced class distribution
2. **Cross-validation**: Multiple split testing
3. **Metric Calculation**: Comprehensive evaluation
4. **Visualization**: Clear result presentation
5. **Statistical Analysis**: Significance testing

## Extensions and Future Work

### Potential Improvements
1. **Attention Fusion**: Replace concatenation with attention mechanisms
2. **Temporal Modeling**: LSTM/GRU for video sequence modeling
3. **Multi-task Learning**: Predict multiple movie attributes
4. **Transfer Learning**: Pre-training on larger movie datasets
5. **Ensemble Methods**: Combine multiple model predictions

### Additional Features
1. **Real-time Prediction**: Live trailer analysis
2. **Web Interface**: User-friendly prediction interface
3. **API Development**: RESTful prediction services
4. **Model Interpretability**: Feature importance analysis
5. **A/B Testing**: Production deployment validation

This comprehensive system provides a robust foundation for movie revenue prediction using state-of-the-art multimodal deep learning techniques. 