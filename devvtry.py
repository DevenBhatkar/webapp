import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from tqdm import tqdm
import json
from torch.utils.data.sampler import SubsetRandomSampler
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

def pad_spectrogram(spec, max_len=100):
    """Pad or truncate spectrogram to fixed length"""
    if spec.shape[0] > max_len:
        return spec[:max_len, :]
    elif spec.shape[0] < max_len:
        pad_amount = max_len - spec.shape[0]
        return F.pad(spec, (0, 0, 0, pad_amount), "constant", 0)
    return spec

# Improved preprocessing function
def preprocess_spectrogram(spectrogram):
    """Apply robust preprocessing to spectrogram"""
    # Convert to float tensor if needed
    if not isinstance(spectrogram, torch.Tensor):
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
    
    # Simple min-max scaling to range [0, 1]
    min_val = torch.min(spectrogram)
    max_val = torch.max(spectrogram)
    if max_val > min_val:  # Avoid division by zero
        spectrogram = (spectrogram - min_val) / (max_val - min_val)
    
    return spectrogram

# Custom Dataset class with robust preprocessing
class SpectrogramDataset(Dataset):
    def __init__(self, spectrogram_dir, csv_path, max_len=100, cache_size=100):
        self.spectrogram_dir = spectrogram_dir
        self.data = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
        
        # Select features for training
        feature_columns = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        self.csv_features = self.scaler.fit_transform(self.data[feature_columns])
        
        # Encode categorical target values
        self.targets = self.label_encoder.fit_transform(self.data['expert_consensus'])
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(self.targets)}")
        
        # Get list of parquet files with caching
        self.spectrogram_files = {
            row['spectrogram_id']: os.path.join(spectrogram_dir, f"{row['spectrogram_id']}.parquet")
            for _, row in self.data.iterrows()
        }
        
        # Initialize cache
        self.cache = {}
        self.cache_size = cache_size
    
    def _calculate_class_weights(self):
        """Calculate class weights to handle imbalanced data"""
        class_sample_count = np.bincount(self.targets)
        total_samples = len(self.targets)
        weight_per_class = total_samples / (len(class_sample_count) * class_sample_count)
        # Normalize weights
        weight_per_class = weight_per_class / weight_per_class.sum() * len(class_sample_count)
        return torch.FloatTensor(weight_per_class)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram_id = self.data.iloc[idx]['spectrogram_id']
        
        # Try to get from cache first
        if spectrogram_id in self.cache:
            spectrogram = self.cache[spectrogram_id]
        else:
            # Load and process spectrogram
            try:
                spec_file = self.spectrogram_files[spectrogram_id]
                spectrogram = pd.read_parquet(spec_file).values
                
                # Apply preprocessing
                spectrogram = preprocess_spectrogram(spectrogram)
                
                # Pad to fixed size
                spectrogram = pad_spectrogram(spectrogram, self.max_len)
                
                # Cache if space available
                if len(self.cache) < self.cache_size:
                    self.cache[spectrogram_id] = spectrogram
            except Exception as e:
                print(f"Error loading spectrogram {spectrogram_id}: {e}")
                # Return a zero tensor in case of loading failure
                spectrogram = torch.zeros((self.max_len, 401))
        
        # Get CSV features and target
        csv_features = torch.FloatTensor(self.csv_features[idx])
        target = torch.LongTensor([self.targets[idx]]).squeeze()
        
        return spectrogram, csv_features, target

# Much simpler CNN model - no LSTM
class SimpleModel(nn.Module):
    def __init__(self, input_size=6, num_classes=6):
        super(SimpleModel, self).__init__()
        
        # Even simpler CNN layers to avoid NaN
        self.features = nn.Sequential(
            # First conv block - smaller filters
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),  # LeakyReLU is more stable
            
            # Second conv block
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            
            # Third conv block
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # Global average pooling to avoid dimension issues
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 + input_size, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x_spec, x_csv):
        batch_size = x_spec.size(0)
        
        # Add channel dimension if needed
        if len(x_spec.shape) == 3:
            x_spec = x_spec.unsqueeze(1)
        
        # Apply CNN
        x = self.features(x_spec)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Concatenate with CSV features
        x = torch.cat([x, x_csv], dim=1)
        
        # Final classification
        x = self.classifier(x)
        return x

class ModelWrapper:
    def __init__(self, model, label_encoder, scaler):
        self.model = model
        self.label_encoder = label_encoder
        self.scaler = scaler
    
    def predict(self, spectrogram, csv_features):
        """Make prediction for web app backend"""
        self.model.eval()
        with torch.no_grad():
            # Preprocess spectrogram
            if isinstance(spectrogram, np.ndarray):
                spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
            
            # Apply the same preprocessing as in training
            spectrogram = preprocess_spectrogram(spectrogram)
            spectrogram = pad_spectrogram(spectrogram)
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dimension
            
            # Preprocess CSV features
            if isinstance(csv_features, (list, np.ndarray)):
                csv_features = np.array(csv_features).reshape(1, -1)
                csv_features = self.scaler.transform(csv_features)
            csv_features = torch.FloatTensor(csv_features)
            
            # Make prediction
            outputs = self.model(spectrogram, csv_features)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            return {
                'predicted_class': predicted_class,
                'probabilities': {
                    cls: float(prob)
                    for cls, prob in zip(self.label_encoder.classes_, probabilities[0].numpy())
                }
            }

class TrainingProgress:
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0,
            'best_epoch': 0
        }
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        if val_acc > self.history['best_val_acc']:
            self.history['best_val_acc'] = val_acc
            self.history['best_epoch'] = epoch
    
    def save(self, filename='training_history.json'):
        with open(filename, 'w') as f:
            json.dump(self.history, f)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, progress_tracker):
    scaler = GradScaler()  # For mixed precision training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_acc = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for spectrograms, csv_features, targets in train_pbar:
            spectrograms = spectrograms.to(device)
            csv_features = csv_features.to(device)
            targets = targets.to(device).squeeze()
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                outputs = model(spectrograms, csv_features)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for spectrograms, csv_features, targets in val_pbar:
                spectrograms = spectrograms.to(device)
                csv_features = csv_features.to(device)
                targets = targets.to(device).squeeze()
                
                outputs = model(spectrograms, csv_features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update progress tracker
        progress_tracker.update(epoch + 1, train_loss, train_acc, val_loss, val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
        
        # Early stopping check
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print('Learning rate too small - stopping training!')
            break
    
    return best_val_acc, best_val_loss

def main():
    # Hyperparameters
    batch_size = 4  # Smaller for stability
    num_epochs = 20
    learning_rate = 0.0001  # Lower learning rate
    max_samples = 1000  # Limit to 1000 samples (set to None to use all data)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset with robust preprocessing
    dataset = SpectrogramDataset(
        spectrogram_dir='train_spectrograms',
        csv_path='train.csv',
        max_len=100,
        cache_size=100
    )
    
    # Split dataset with a fixed seed for reproducibility
    dataset_size = len(dataset)
    print(f"Total available samples: {dataset_size}")
    
    # Limit the number of samples if specified
    if max_samples and max_samples < dataset_size:
        print(f"Limiting to {max_samples} samples for faster training")
        # Use a fixed seed for reproducible sampling
        np.random.seed(42)
        all_indices = list(range(dataset_size))
        np.random.shuffle(all_indices)
        selected_indices = all_indices[:max_samples]
        dataset_size = max_samples
    else:
        print(f"Using all {dataset_size} samples for training")
        selected_indices = list(range(dataset_size))
    
    # Split into train/validation
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(selected_indices)
    train_indices, val_indices = selected_indices[split:], selected_indices[:split]
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=False
    )
    
    # Create simplified model
    num_classes = len(dataset.label_encoder.classes_)
    model = SimpleModel(input_size=6, num_classes=num_classes).to(device)
    print("\nModel Architecture:")
    print(model)
    
    # Test forward pass with a single batch
    print("\nTesting a single batch...")
    try:
        test_batch = next(iter(train_loader))
        specs, features, targets = test_batch
        print(f"Spectrogram shape: {specs.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        specs = specs.to(device)
        features = features.to(device)
        outputs = model(specs, features)
        print(f"Model output shape: {outputs.shape}")
        print("Single batch test successful!")
    except Exception as e:
        print(f"Error during test batch: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit if test fails
    
    # Loss function with class weights
    class_weights = dataset.class_weights.to(device)
    print(f"Class weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer - simple SGD with momentum is more stable than Adam
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training history
    best_val_acc = 0
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0
    }
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (spectrograms, csv_features, targets) in enumerate(train_loader):
            try:
                spectrograms = spectrograms.to(device)
                csv_features = csv_features.to(device)
                targets = targets.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(spectrograms, csv_features)
                loss = criterion(outputs, targets)
                
                # Check for NaN loss
                if torch.isnan(loss).any():
                    print(f"NaN loss detected in batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                train_loss += loss.item()
                
                # Print progress every few batches
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | Acc: {100.*train_correct/train_total:.2f}%")
                
                # Memory cleanup
                del spectrograms, csv_features, targets, outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        if train_total > 0:
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
        else:
            print("Warning: No valid training samples in this epoch")
            train_loss = float('inf')
            train_acc = 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for spectrograms, csv_features, targets in val_loader:
                try:
                    spectrograms = spectrograms.to(device)
                    csv_features = csv_features.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(spectrograms, csv_features)
                    loss = criterion(outputs, targets)
                    
                    # Skip NaN losses
                    if torch.isnan(loss).any():
                        continue
                    
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    val_loss += loss.item()
                    
                    # Memory cleanup
                    del spectrograms, csv_features, targets, outputs, loss
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if val_total > 0:
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Learning rate scheduler step
            scheduler.step(val_loss)
        else:
            print("Warning: No valid validation samples in this epoch")
            val_loss = float('inf')
            val_acc = 0
        
        # Record history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Early stopping if learning rate gets too small
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate is too small - stopping training")
            break
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Save model wrapper
    model_wrapper = ModelWrapper(model, dataset.label_encoder, dataset.scaler)
    with open('model_wrapper.pkl', 'wb') as f:
        pickle.dump(model_wrapper, f)
    
    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best performance achieved at epoch {history['best_epoch']}")
    print("\nModel saved as 'model_wrapper.pkl' for web app use")
    print("Training history saved as 'training_history.json'")
    
    # Instructions for full training
    print("\nTo train on all data later, set max_samples=None in the main function")

if __name__ == '__main__':
    main()
