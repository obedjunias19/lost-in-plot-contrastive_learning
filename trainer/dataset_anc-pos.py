import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample


class FilmPairDataset(Dataset):
    """
    Dataset for contrastive learning on films.
    
    For each anchor film, samples a positive (same theme) and negatives (different themes).
    Returns pairs suitable for InfoNCE loss with batch negatives.
    """
    def __init__(self, df, encoders, tokenizer=None, max_length=512):
        """
        Args:
            df: DataFrame with columns: combined_text_baseline, genre, theme, era, director
            encoders: Dict of sklearn LabelEncoders for each dimension
            tokenizer: Optional tokenizer (not used here, kept for compatibility)
            max_length: Maximum sequence length (kept for compatibility)
        """
        self.df = df
        self.texts = df["combined_text_baseline"].tolist()
        
        # Encode labels
        self.genre_labels = encoders['genre'].transform(df["genre"])
        self.theme_labels = encoders['theme'].transform(df["theme"])
        self.era_labels = encoders['era'].transform(df["era"])
        self.director_labels = encoders['director'].transform(df["director"])
        
        # Multi-hot encoding for genres (could be multiple per movie)
        self.num_genres = len(encoders['genre'].classes_)
        self.genre_multihot = self._create_multihot_genre(df["genre"])
        
        # Build index: theme -> list of indices with that theme
        unique_themes = np.unique(self.theme_labels)
        self.theme_index = {
            theme: np.where(self.theme_labels == theme)[0].tolist()
            for theme in unique_themes
        }
        
        # Filter themes with only 1 sample (can't form positive pairs)
        self.valid_themes = [t for t, indices in self.theme_index.items() if len(indices) >= 2]
    
    def _create_multihot_genre(self, genre_col):
        """Convert genre labels to multi-hot encoding."""
        multihot = np.zeros((len(genre_col), self.num_genres))
        for i, genre_label in enumerate(self.genre_labels):
            multihot[i, genre_label] = 1.0
        return multihot
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dict with:
            - 'anchor_text': Movie text
            - 'positive_text': Different movie with same theme
            - 'genre_labels': Multi-hot genre vector
            - 'theme_labels': Theme label (single-label)
            - 'era_labels': Era/decade label
        """
        anchor_text = self.texts[idx]
        anchor_theme = self.theme_labels[idx]
        
        # Sample positive: different movie with same theme
        if anchor_theme in self.valid_themes:
            positive_candidates = [i for i in self.theme_index[anchor_theme] if i != idx]
            positive_idx = np.random.choice(positive_candidates)
            positive_text = self.texts[positive_idx]
        else:
            # Fallback: use anchor itself if no valid positive
            positive_text = anchor_text
        
        return {
            'anchor_text': anchor_text,
            'positive_text': positive_text,
            'genre_labels': torch.from_numpy(self.genre_multihot[idx]).float(),
            'theme_labels': torch.tensor(self.theme_labels[idx], dtype=torch.long),
            'era_labels': torch.tensor(self.era_labels[idx], dtype=torch.long)
        }


def create_evaluation_pairs_for_dimension(dataset, dimension):
    """
    Create positive and negative pairs for evaluating a specific dimension.
    
    Returns list of InputExample objects suitable for sentence-transformers evaluation.
    """
    eval_examples = []
    texts = dataset.texts
    
    if dimension == 'theme':
        labels = dataset.theme_labels
    elif dimension == 'era':
        labels = dataset.era_labels
    else:
        raise ValueError(f"Dimension {dimension} not supported")
    
    unique_labels = np.unique(labels)
    
    # Positive pairs: same label
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) >= 2:
            for _ in range(min(10, len(indices))):
                i, j = np.random.choice(indices, 2, replace=False)
                eval_examples.append(InputExample(
                    texts=[texts[i], texts[j]], 
                    label=1.0
                ))
    
    # Negative pairs: different labels
    for _ in range(len(eval_examples)):
        label1, label2 = np.random.choice(unique_labels, 2, replace=False)
        idx1 = np.random.choice(np.where(labels == label1)[0])
        idx2 = np.random.choice(np.where(labels == label2)[0])
        eval_examples.append(InputExample(
            texts=[texts[idx1], texts[idx2]], 
            label=0.0
        ))
    
    return eval_examples