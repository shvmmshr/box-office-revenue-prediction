"""
Simplified Multimodal Movie Revenue Prediction Trainer
This version focuses on the core architecture and training process without requiring
actual video downloads, making it easier to test and develop.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json

class Config:
    """Configuration class for model parameters"""
    # Data splits
    SPLIT_OPTIONS = {
        'option1': (0.7, 0.2, 0.1),  # 70-20-10
        'option2': (0.75, 0.15, 0.1), # 75-15-10
        'option3': (0.8, 0.1, 0.1)    # 80-10-10
    }
    
    # Model parameters
    MAX_TEXT_LENGTH = 512
    VIDEO_FRAME_SIZE = 224
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 30
    
    # Training parameters
    BATCH_SIZE = 8  # Reduced for easier testing
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10  # Reduced for testing
    PATIENCE = 5
    
    # Model dimensions
    TEXT_EMBEDDING_DIM = 768
    VIDEO_EMBEDDING_DIM = 2048
    AUDIO_EMBEDDING_DIM = 1024
    FUSION_DIM = 512
    NUM_CLASSES = 8
    
    # Simulated features
    FRAMES_PER_VIDEO = 30

class TextEncoder(nn.Module):
    """BERT-based text encoder for movie plots/synopsis"""
    
    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(Config.TEXT_EMBEDDING_DIM, Config.TEXT_EMBEDDING_DIM)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        encoded = self.fc(pooled_output)
        return F.relu(encoded)

class VideoEncoder(nn.Module):
    """Simulated video encoder using random features for testing"""
    
    def __init__(self):
        super(VideoEncoder, self).__init__()
        # Simulate video feature extraction with FC layers
        self.fc1 = nn.Linear(Config.FRAMES_PER_VIDEO * 1000, 1024)  # Simulated frame features
        self.fc2 = nn.Linear(1024, Config.VIDEO_EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, video_features):
        # video_features: (batch_size, simulated_features)
        x = F.relu(self.fc1(video_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class AudioEncoder(nn.Module):
    """Simulated audio encoder using random features for testing"""
    
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # Simulate audio feature extraction
        audio_length = Config.AUDIO_SAMPLE_RATE * Config.AUDIO_DURATION
        self.fc1 = nn.Linear(1000, 512)  # Simulated audio features
        self.fc2 = nn.Linear(512, Config.AUDIO_EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, audio_features):
        # audio_features: (batch_size, simulated_features)
        x = F.relu(self.fc1(audio_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model combining text, video, and audio"""
    
    def __init__(self):
        super(MultimodalFusionModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        # Fusion layers
        total_dim = Config.TEXT_EMBEDDING_DIM + Config.VIDEO_EMBEDDING_DIM + Config.AUDIO_EMBEDDING_DIM
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, Config.FUSION_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, Config.FUSION_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM // 2, Config.NUM_CLASSES)
        )
    
    def forward(self, text_input_ids, text_attention_mask, video_features, audio_features):
        # Encode each modality
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        video_features = self.video_encoder(video_features)
        audio_features = self.audio_encoder(audio_features)
        
        # Concatenate features
        combined_features = torch.cat([text_features, video_features, audio_features], dim=1)
        
        # Apply fusion layers
        output = self.fusion_layers(combined_features)
        return output

class SimplifiedMovieDataset(Dataset):
    """Simplified dataset with simulated video/audio features"""
    
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text processing
        text = str(row['Description']) if pd.notna(row['Description']) else ""
        text_encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=Config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        # Simulated video features (in practice, these would come from actual video processing)
        video_features = torch.randn(Config.FRAMES_PER_VIDEO * 1000)
        
        # Simulated audio features (in practice, these would come from actual audio processing)
        audio_features = torch.randn(1000)
        
        # Target
        target = row['y'] if 'y' in row else 0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'video_features': video_features,
            'audio_features': audio_features,
            'target': torch.LongTensor([target])[0]
        }

class ModelTrainer:
    """Training and evaluation utilities"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            video_features = batch['video_features'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            target = batch['target'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, video_features, audio_features)
            loss = criterion(outputs, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(targets, predictions)
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                video_features = batch['video_features'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, video_features, audio_features)
                loss = criterion(outputs, target)
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.calculate_metrics(targets, predictions)
        return avg_loss, metrics, predictions, targets
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        return cm

def prepare_data(data_path, split_option='option1'):
    """Prepare and split the dataset"""
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Description', 'Verdict'])
    
    # Take a smaller sample for testing
    df = df.head(1000)  # Use first 1000 rows for testing
    
    # Label mapping
    label_mapping = {
        'Disaster': 0, 'Flop': 1, 'Successful': 2, 'Average': 3,
        'Hit': 4, 'Outstanding': 5, 'Superhit': 6, 'Blockbuster': 7
    }
    
    df['y'] = df['Verdict'].map(label_mapping)
    df = df.dropna(subset=['y'])
    
    # Get split ratios
    train_ratio, val_ratio, test_ratio = Config.SPLIT_OPTIONS[split_option]
    
    # Split data
    X = df.drop(['y', 'Verdict'], axis=1)
    y = df['y']
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), 
        random_state=42, stratify=y
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size),
        random_state=42, stratify=y_temp
    )
    
    # Combine X and y back
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"Data split ({split_option}):")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df, label_mapping

def create_data_loaders(train_df, val_df, test_df, tokenizer):
    """Create PyTorch data loaders"""
    
    train_dataset = SimplifiedMovieDataset(train_df, tokenizer)
    val_dataset = SimplifiedMovieDataset(val_df, tokenizer)
    test_dataset = SimplifiedMovieDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=0)  # num_workers=0 for easier debugging
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, split_name):
    """Train the multimodal model"""
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    trainer = ModelTrainer(model, device)
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training for {split_name}...")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validation
        val_loss, val_metrics, _, _ = trainer.evaluate(val_loader, criterion)
        val_acc = val_metrics['accuracy']
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val F1 (weighted): {val_metrics['f1_weighted']:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{split_name}.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title(f'Loss - {split_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title(f'Accuracy - {split_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{split_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return trainer

def evaluate_model(trainer, test_loader, label_mapping, split_name):
    """Comprehensive model evaluation"""
    
    print(f"\nEvaluating model on test set for {split_name}...")
    
    # Load best model
    trainer.model.load_state_dict(torch.load(f'best_model_{split_name}.pth'))
    
    # Evaluate
    test_loss, test_metrics, predictions, targets = trainer.evaluate(test_loader, nn.CrossEntropyLoss())
    
    # Print metrics
    print(f"\n{'='*50}")
    print(f"TEST SET EVALUATION RESULTS - {split_name}")
    print(f"{'='*50}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"F1 Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted): {test_metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {test_metrics['recall_weighted']:.4f}")
    
    # Classification report
    class_names = list(label_mapping.keys())
    print("\nDetailed Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = trainer.plot_confusion_matrix(targets, predictions, class_names)
    
    # Save results
    results = {
        'split_name': split_name,
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'targets': targets,
        'classification_report': classification_report(targets, predictions, 
                                                     target_names=class_names, 
                                                     output_dict=True, zero_division=0)
    }
    
    return results

def main():
    """Main execution function"""
    
    print("Simplified Multimodal Movie Revenue Prediction System")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    print("\nInitializing components...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Results storage
    all_results = {}
    
    # Test all three split options
    for split_option in ['option1', 'option2', 'option3']:
        print(f"\n{'='*30} {split_option.upper()} {'='*30}")
        
        # Prepare data
        train_df, val_df, test_df, label_mapping = prepare_data('Data/TMRDB.csv', split_option)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, tokenizer
        )
        
        # Initialize model
        model = MultimodalFusionModel()
        print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        trainer = train_model(model, train_loader, val_loader, device, split_option)
        
        # Evaluate model
        results = evaluate_model(trainer, test_loader, label_mapping, split_option)
        all_results[split_option] = results
        
        print(f"\nCompleted {split_option}")
    
    # Compare results across splits
    print(f"\n{'='*70}")
    print("COMPARISON ACROSS DIFFERENT DATA SPLITS")
    print(f"{'='*70}")
    
    comparison_data = []
    for split_option, results in all_results.items():
        ratios = Config.SPLIT_OPTIONS[split_option]
        comparison_data.append({
            'Split': f"{ratios[0]*100:.0f}-{ratios[1]*100:.0f}-{ratios[2]*100:.0f}",
            'Accuracy': f"{results['test_metrics']['accuracy']:.4f}",
            'F1 (Weighted)': f"{results['test_metrics']['f1_weighted']:.4f}",
            'F1 (Macro)': f"{results['test_metrics']['f1_macro']:.4f}",
            'Precision (W)': f"{results['test_metrics']['precision_weighted']:.4f}",
            'Recall (W)': f"{results['test_metrics']['recall_weighted']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comparison and all results
    comparison_df.to_csv('split_comparison.csv', index=False)
    
    with open('all_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for k, v in all_results.items():
            serializable_results[k] = {
                'split_name': v['split_name'],
                'test_metrics': v['test_metrics'],
                'confusion_matrix': v['confusion_matrix'],
                'predictions': [int(x) for x in v['predictions']],
                'targets': [int(x) for x in v['targets']],
                'classification_report': v['classification_report']
            }
        json.dump(serializable_results, f, indent=2)
    
    print("\nTraining completed! Generated files:")
    print("- best_model_[split].pth: Best model weights for each split")
    print("- training_curves_[split].png: Training visualization for each split")
    print("- confusion_matrix.png: Confusion matrix")
    print("- split_comparison.csv: Comparison across different splits")
    print("- all_results.json: Detailed results for all splits")

if __name__ == "__main__":
    main() 