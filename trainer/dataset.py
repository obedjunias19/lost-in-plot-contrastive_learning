import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample

class MultiTaskFilmDataset(Dataset):
    def __init__(self, df, encoders, tokenizer=None, max_length=512):
        self.df = df
        self.texts = df["combined_text_baseline"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Store encodings for each dimension
        self.genre_labels = encoders['genre'].transform(df["genre"])
        self.theme_labels = encoders['theme'].transform(df["theme"])
        self.era_labels = encoders['era'].transform(df["era"])  # Could be decade or period
        self.director_labels = encoders['director'].transform(df["director"])
        
        # Create mapping dictionaries for each dimension
        self.label_maps = {}
        for dim in ['genre', 'theme', 'era', 'director']:
            labels = getattr(self, f"{dim}_labels")
            unique_labels = np.unique(labels)
            self.label_maps[dim] = {
                label: np.where(labels == label)[0] 
                for label in unique_labels
            }
            
            # Filter labels with only one example
            setattr(self, f"valid_{dim}_labels", 
                    [label for label, indices in self.label_maps[dim].items() 
                     if len(indices) > 1])
    
    def __len__(self):
        return len(self.df) * 3  # Generate multiple triplets per data point
    
    def __getitem__(self, idx):
        actual_idx = idx % len(self.df)
        dimension = ['genre', 'theme', 'era', 'director'][idx % 4]  # Cycle through dimensions
        
        anchor_text = self.texts[actual_idx]
        anchor_label = getattr(self, f"{dimension}_labels")[actual_idx]
        valid_labels = getattr(self, f"valid_{dimension}_labels")
        label_map = self.label_maps[dimension]
        
        # Positive sample (same dimension value)
        if anchor_label in valid_labels:
            positive_indices = label_map[anchor_label]
            positive_indices = positive_indices[positive_indices != actual_idx]
            if len(positive_indices) > 0:
                positive_idx = np.random.choice(positive_indices)
                positive_text = self.texts[positive_idx]
            else:
                positive_text = anchor_text
        else:
            positive_text = anchor_text
        
        # Negative sample (different dimension value)
        negative_candidates = [l for l in valid_labels if l != anchor_label]
        if negative_candidates:
            negative_label = np.random.choice(negative_candidates)
            negative_idx = np.random.choice(label_map[negative_label])
            negative_text = self.texts[negative_idx]
        else:
            negative_idx = np.random.choice(range(len(self.texts)))
            while negative_idx == actual_idx:
                negative_idx = np.random.choice(range(len(self.texts)))
            negative_text = self.texts[negative_idx]
        
        # Return triplet with dimension info
        return {
            'texts': [anchor_text, positive_text, negative_text],
            'dimension': dimension
        }

def create_evaluation_pairs_for_dimension(dataset, dimension):
    """Create evaluation pairs for a specific dimension"""
    eval_examples = []
    
    # Get texts
    texts = dataset.texts
    # Get labels for this dimension
    labels = getattr(dataset, f"{dimension}_labels")
    
    # Create positive pairs
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) >= 2:
            for _ in range(min(10, len(indices))):
                i, j = np.random.choice(indices, 2, replace=False)
                eval_examples.append(InputExample(texts=[texts[i], texts[j]], label=1.0))
    
    # Create negative pairs
    for _ in range(len(eval_examples)):
        label1, label2 = np.random.choice(unique_labels, 2, replace=False)
        idx1 = np.random.choice(np.where(labels == label1)[0])
        idx2 = np.random.choice(np.where(labels == label2)[0])
        eval_examples.append(InputExample(texts=[texts[idx1], texts[idx2]], label=0.0))
    
    return eval_examples