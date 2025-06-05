"""
Multimodal Movie Revenue Prediction System
This system combines text analysis (plot/synopsis) and video analysis (YouTube trailers)
to predict movie revenue categories using deep learning models.

Revenue Categories:
- Disaster (0), Flop (1), Successful (2), Average (3)
- Hit (4), Outstanding (5), Superhit (6), Blockbuster (7)
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import cv2
import librosa
import pytube
from pytube import YouTube
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                           confusion_matrix, classification_report, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Models
import torchvision
from torchvision import transforms, models
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

# Audio processing
import torchaudio
from torchaudio import transforms as audio_transforms

# Additional utilities
from tqdm import tqdm
import pickle
import json
from urllib.parse import urlparse, parse_qs
import re
import time

class Config:
    """Configuration class for model parameters"""
    # Data splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Alternative splits
    SPLIT_OPTIONS = {
        'option1': (0.7, 0.2, 0.1),  # 70-20-10
        'option2': (0.75, 0.15, 0.1), # 75-15-10
        'option3': (0.8, 0.1, 0.1)    # 80-10-10
    }
    
    # Model parameters
    MAX_TEXT_LENGTH = 512
    VIDEO_FRAME_SIZE = 224
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 30  # seconds
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10
    
    # Model dimensions
    TEXT_EMBEDDING_DIM = 768  # BERT-base
    VIDEO_EMBEDDING_DIM = 2048  # ResNet50
    AUDIO_EMBEDDING_DIM = 1024
    FUSION_DIM = 512
    NUM_CLASSES = 8
    
    # Video processing
    FRAMES_PER_VIDEO = 30
    VIDEO_DURATION = 60  # seconds to analyze

class YouTubeVideoProcessor:
    """Handles YouTube video downloading and processing"""
    
    def __init__(self, output_dir='temp_videos'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_video_id(self, url):
        """Extract YouTube video ID from URL"""
        if 'youtube.com/watch' in url:
            return parse_qs(urlparse(url).query).get('v', [None])[0]
        elif 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
        elif re.match(r'^[A-Za-z0-9_-]{11}$', url):
            return url
        return None
    
    def download_video(self, video_url, video_id):
        """Download YouTube video"""
        try:
            yt = YouTube(video_url)
            stream = yt.streams.filter(file_extension='mp4', res='720p').first()
            if not stream:
                stream = yt.streams.filter(file_extension='mp4').first()
            
            video_path = os.path.join(self.output_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                stream.download(output_path=self.output_dir, filename=f"{video_id}.mp4")
            return video_path
        except Exception as e:
            print(f"Error downloading video {video_id}: {e}")
            return None
    
    def extract_frames(self, video_path, num_frames=30):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Extract frames evenly distributed across the video
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            return np.array(frames)
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return None
    
    def extract_audio(self, video_path):
        """Extract audio from video"""
        try:
            # Extract audio using librosa
            audio, sr = librosa.load(video_path, sr=Config.AUDIO_SAMPLE_RATE, 
                                   duration=Config.AUDIO_DURATION)
            return audio, sr
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None, None

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
    """CNN-based video encoder for trailer frames"""
    
    def __init__(self):
        super(VideoEncoder, self).__init__()
        # Use pre-trained ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        
        # Temporal pooling and processing
        self.temporal_conv = nn.Conv1d(Config.VIDEO_EMBEDDING_DIM, 
                                     Config.VIDEO_EMBEDDING_DIM, 
                                     kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(Config.VIDEO_EMBEDDING_DIM, Config.VIDEO_EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, frames):
        # frames: (batch_size, num_frames, 3, height, width)
        batch_size, num_frames = frames.shape[:2]
        
        # Reshape for CNN processing
        frames = frames.view(-1, 3, Config.VIDEO_FRAME_SIZE, Config.VIDEO_FRAME_SIZE)
        
        # Extract features for each frame
        frame_features = self.backbone(frames)  # (batch*num_frames, 2048)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Temporal processing
        frame_features = frame_features.transpose(1, 2)  # (batch, features, frames)
        temporal_features = self.temporal_conv(frame_features)
        pooled_features = self.global_pool(temporal_features).squeeze(-1)
        
        # Final encoding
        encoded = self.fc(pooled_features)
        encoded = self.dropout(encoded)
        return F.relu(encoded)

class AudioEncoder(nn.Module):
    """CNN-based audio encoder for trailer audio"""
    
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # 1D CNN for audio processing
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1024, stride=512)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=64, stride=32)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=32, stride=16)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=16, stride=8)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, Config.AUDIO_EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, audio):
        # audio: (batch_size, audio_length)
        audio = audio.unsqueeze(1)  # (batch_size, 1, audio_length)
        
        x = F.relu(self.conv1(audio))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.dropout(x)
        return F.relu(x)

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
        
        # Attention mechanism for modality fusion
        self.attention = nn.MultiheadAttention(embed_dim=Config.FUSION_DIM, num_heads=8)
        self.modality_projection = nn.Linear(total_dim, Config.FUSION_DIM)
    
    def forward(self, text_input_ids, text_attention_mask, video_frames, audio):
        # Encode each modality
        text_features = self.text_encoder(text_input_ids, text_attention_mask)
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio)
        
        # Concatenate features
        combined_features = torch.cat([text_features, video_features, audio_features], dim=1)
        
        # Apply fusion layers
        output = self.fusion_layers(combined_features)
        return output

class MovieDataset(Dataset):
    """Dataset class for movie data with multimodal inputs"""
    
    def __init__(self, dataframe, tokenizer, video_processor, transform=None, mode='train'):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.transform = transform
        self.mode = mode
        
        # Video transforms
        self.video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.VIDEO_FRAME_SIZE, Config.VIDEO_FRAME_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
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
        
        # Video processing
        video_id = self.video_processor.extract_video_id(row['Trailer'])
        frames = np.zeros((Config.FRAMES_PER_VIDEO, Config.VIDEO_FRAME_SIZE, 
                          Config.VIDEO_FRAME_SIZE, 3))
        audio = np.zeros(Config.AUDIO_SAMPLE_RATE * Config.AUDIO_DURATION)
        
        if video_id and self.mode == 'train':  # Only process videos during training
            try:
                video_path = self.video_processor.download_video(row['Trailer'], video_id)
                if video_path:
                    extracted_frames = self.video_processor.extract_frames(video_path, Config.FRAMES_PER_VIDEO)
                    extracted_audio, _ = self.video_processor.extract_audio(video_path)
                    
                    if extracted_frames is not None:
                        frames = extracted_frames
                    if extracted_audio is not None:
                        audio = extracted_audio[:len(audio)]  # Truncate to desired length
            except:
                pass  # Use zero frames/audio if processing fails
        
        # Transform video frames
        transformed_frames = []
        for frame in frames:
            if frame.max() > 1:  # If pixel values are in [0, 255]
                frame = frame.astype(np.uint8)
            else:  # If pixel values are in [0, 1]
                frame = (frame * 255).astype(np.uint8)
            transformed_frame = self.video_transform(frame)
            transformed_frames.append(transformed_frame)
        
        video_tensor = torch.stack(transformed_frames)
        audio_tensor = torch.FloatTensor(audio)
        
        # Target
        target = row['y'] if 'y' in row else 0
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'video_frames': video_tensor,
            'audio': audio_tensor,
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
            video_frames = batch['video_frames'].to(self.device)
            audio = batch['audio'].to(self.device)
            target = batch['target'].to(self.device)
            
            outputs = self.model(input_ids, attention_mask, video_frames, audio)
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
                video_frames = batch['video_frames'].to(self.device)
                audio = batch['audio'].to(self.device)
                target = batch['target'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, video_frames, audio)
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
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
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
        plt.show()
        return cm

def prepare_data(data_path, split_option='option1'):
    """Prepare and split the dataset"""
    print("Loading and preparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Description', 'Trailer', 'Verdict'])
    
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

def create_data_loaders(train_df, val_df, test_df, tokenizer, video_processor):
    """Create PyTorch data loaders"""
    
    train_dataset = MovieDataset(train_df, tokenizer, video_processor, mode='train')
    val_dataset = MovieDataset(val_df, tokenizer, video_processor, mode='val')
    test_dataset = MovieDataset(test_df, tokenizer, video_processor, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device):
    """Train the multimodal model"""
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    trainer = ModelTrainer(model, device)
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("Starting training...")
    
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
        print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_multimodal_model.pth')
            print("Saved best model!")
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([scheduler.get_last_lr()[0]] * len(train_losses))
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return trainer

def evaluate_model(trainer, test_loader, label_mapping):
    """Comprehensive model evaluation"""
    
    print("\nEvaluating model on test set...")
    
    # Load best model
    trainer.model.load_state_dict(torch.load('best_multimodal_model.pth'))
    
    # Evaluate
    test_loss, test_metrics, predictions, targets = trainer.evaluate(test_loader, nn.CrossEntropyLoss())
    
    # Print metrics
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1 Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"F1 Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Precision (Weighted): {test_metrics['precision_weighted']:.4f}")
    print(f"Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"Recall (Weighted): {test_metrics['recall_weighted']:.4f}")
    print(f"Recall (Macro): {test_metrics['recall_macro']:.4f}")
    
    # Classification report
    class_names = list(label_mapping.keys())
    print("\nDetailed Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = trainer.plot_confusion_matrix(targets, predictions, class_names)
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'targets': targets,
        'classification_report': classification_report(targets, predictions, 
                                                     target_names=class_names, 
                                                     output_dict=True)
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation results saved to 'evaluation_results.json'")
    
    return results

def main():
    """Main execution function"""
    
    print("Multimodal Movie Revenue Prediction System")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    print("\nInitializing components...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    video_processor = YouTubeVideoProcessor()
    
    # Prepare data for all three split options
    split_results = {}
    
    for split_option in ['option1', 'option2', 'option3']:
        print(f"\n{'='*20} {split_option.upper()} {'='*20}")
        
        # Prepare data
        train_df, val_df, test_df, label_mapping = prepare_data('Data/TMRDB.csv', split_option)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_df, val_df, test_df, tokenizer, video_processor
        )
        
        # Initialize model
        model = MultimodalFusionModel()
        print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train model
        trainer = train_model(model, train_loader, val_loader, device)
        
        # Evaluate model
        results = evaluate_model(trainer, test_loader, label_mapping)
        split_results[split_option] = results
        
        # Save model for this split
        torch.save(model.state_dict(), f'multimodal_model_{split_option}.pth')
        
        print(f"\nCompleted {split_option}")
    
    # Compare results across splits
    print("\n" + "="*60)
    print("COMPARISON ACROSS DIFFERENT DATA SPLITS")
    print("="*60)
    
    comparison_data = []
    for split_option, results in split_results.items():
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
    
    # Save comparison
    comparison_df.to_csv('split_comparison.csv', index=False)
    
    print("\nTraining completed! Check the generated files:")
    print("- best_multimodal_model.pth: Best model weights")
    print("- evaluation_results.json: Detailed evaluation metrics")
    print("- training_curves.png: Training visualization")
    print("- split_comparison.csv: Comparison across different splits")

if __name__ == "__main__":
    main() 