import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sentence_transformers import SentenceTransformer, models
from sklearn.preprocessing import LabelEncoder

from trainer.model import FilmRetrievalModel, MultiTaskFilmLoss
from trainer.dataset import FilmPairDataset


def create_film_model(base_model="bert-base-uncased", embedding_dim=300, device=None, 
                      num_genres=None, num_decades=None, num_themes=None):
    """
    Create dense retrieval model with auxiliary classification heads.
    
    Args:
        base_model: HuggingFace model ID for BERT
        embedding_dim: Dimension of retrieval embeddings
        device: torch device
        num_genres: Number of genre classes
        num_decades: Number of decade classes
        num_themes: Number of theme classes
    
    Returns:
        st_model: SentenceTransformer with attached FilmRetrievalModel
    """
    # BERT + pooling pipeline
    word_embedding_model = models.Transformer(base_model, max_seq_length=512)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode='mean'
    )
    
    input_dim = pooling_model.get_sentence_embedding_dimension()
    
    # Film-specific retrieval and classification heads
    film_model = FilmRetrievalModel(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_genres=num_genres,
        num_decades=num_decades,
        num_themes=num_themes
    )
    
    # Base SentenceTransformer (encoder + pooling, frozen)
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Attach film model for retrieval + classification
    st_model.add_module('film_model', film_model)
    
    # Keep original encode for encoding texts
    original_encode = st_model.encode
    
    def film_encode(sentences, convert_to_tensor=True, batch_size=32, **kwargs):
        """Encode sentences to retrieval embeddings."""
        # Get pooled embeddings (frozen BERT + pooling)
        pooled = original_encode(sentences, convert_to_tensor=True, batch_size=batch_size, **kwargs)
        
        if device is not None:
            pooled = pooled.to(device)
        
        # Project to retrieval space (with L2 norm)
        z_hat = st_model.film_model(pooled, return_heads=False)
        
        return z_hat
    
    st_model.encode = film_encode
    
    # Store reference for training
    st_model.original_encode = original_encode
    
    return st_model


def train_film_model(train_dataset, val_dataset, st_model, output_path, 
                     epochs=8, batch_size=16, lr=2e-5, device=None,
                     w_retr=1.0, w_genre=0.1, w_decade=0.1, w_theme=0.1):
    """
    Train dense retrieval model with multi-task loss.
    
    Args:
        train_dataset: FilmPairDataset for training
        val_dataset: FilmPairDataset for validation
        st_model: SentenceTransformer with FilmRetrievalModel attached
        output_path: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: torch device
        w_retr, w_genre, w_decade, w_theme: Loss weights
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    # Loss function
    loss_fn = MultiTaskFilmLoss(
        tau=0.05,
        w_retr=w_retr,
        w_genre=w_genre,
        w_decade=w_decade,
        w_theme=w_theme
    ).to(device)
    
    # Optimizer: only film_model parameters (freeze BERT)
    params = list(st_model.film_model.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    
    # Scheduler: cosine annealing with warmup
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    
    st_model.to(device)
    st_model.film_model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        st_model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in train_loader:
            anchor_texts = batch['anchor_text']
            positive_texts = batch['positive_text']
            
            # Encode anchor and positive with frozen BERT
            z_anchors = st_model.encode(anchor_texts, convert_to_tensor=True)
            z_positives = st_model.encode(positive_texts, convert_to_tensor=True)
            
            # Forward through projection heads for auxiliary loss
            pooled_anchors = st_model.original_encode(anchor_texts, convert_to_tensor=True)
            pooled_anchors = pooled_anchors.to(device)
            z_hat_aux, aux_outputs = st_model.film_model(pooled_anchors, return_heads=True)
            
            # Prepare batch for loss computation
            loss_batch = {
                'genre_labels': batch['genre_labels'].to(device),
                'theme_labels': batch['theme_labels'].to(device),
                'era_labels': batch['era_labels'].to(device),
                'aux_outputs': {
                    k: v.to(device) for k, v in aux_outputs.items()
                }
            }
            
            # Compute loss
            loss = loss_fn(z_anchors, z_positives, loss_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / max(1, train_steps)
        
        # Validation
        st_model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor_texts = batch['anchor_text']
                positive_texts = batch['positive_text']
                
                z_anchors = st_model.encode(anchor_texts, convert_to_tensor=True)
                z_positives = st_model.encode(positive_texts, convert_to_tensor=True)
                
                pooled_anchors = st_model.original_encode(anchor_texts, convert_to_tensor=True)
                pooled_anchors = pooled_anchors.to(device)
                z_hat_aux, aux_outputs = st_model.film_model(pooled_anchors, return_heads=True)
                
                loss_batch = {
                    'genre_labels': batch['genre_labels'].to(device),
                    'theme_labels': batch['theme_labels'].to(device),
                    'era_labels': batch['era_labels'].to(device),
                    'aux_outputs': {
                        k: v.to(device) for k, v in aux_outputs.items()
                    }
                }
                
                loss = loss_fn(z_anchors, z_positives, loss_batch)
                val_loss += loss.item()
                val_steps += 1
                
                if val_steps >= 20:  # Sampled validation
                    break
        
        avg_val_loss = val_loss / max(1, val_steps) if val_steps > 0 else float('inf')
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_path, exist_ok=True)
            st_model.save(os.path.join(output_path, "best_model"))
    
    return st_model


def extract_embeddings(model, texts, batch_size=32, device=None):
    """
    Extract normalized retrieval embeddings for texts.
    
    Args:
        model: SentenceTransformer with film_encode
        texts: List of text strings
        batch_size: Batch size for encoding
        device: torch device
    
    Returns:
        Numpy array of shape (len(texts), embedding_dim)
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.eval()
    
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    
    return embeddings.cpu().numpy()


def load_data(data_path):
    """Load and preprocess film dataset."""
    df = pd.read_csv(data_path)
    
    # Create label encoders
    encoders = {}
    dimensions = ['genre', 'theme', 'era', 'director']
    
    for dim in dimensions:
        encoder = LabelEncoder()
        encoder.fit(df[dim].fillna('unknown'))
        encoders[dim] = encoder
    
    # Train/val split
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    return train_df, val_df, encoders


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df, val_df, encoders = load_data(args.data_path)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FilmPairDataset(train_df, encoders)
    val_dataset = FilmPairDataset(val_df, encoders)
    
    # Count classes
    num_genres = len(encoders['genre'].classes_)
    num_decades = len(encoders['era'].classes_)
    num_themes = len(encoders['theme'].classes_)
    
    print(f"Num genres: {num_genres}, decades: {num_decades}, themes: {num_themes}")
    
    # Create model
    print("Creating model...")
    model = create_film_model(
        base_model=args.base_model,
        embedding_dim=args.embedding_dim,
        device=device,
        num_genres=num_genres,
        num_decades=num_decades,
        num_themes=num_themes
    )
    
    # Train
    print("Training model...")
    model = train_film_model(
        train_dataset,
        val_dataset,
        model,
        output_path=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        w_retr=args.w_retr,
        w_genre=args.w_genre,
        w_decade=args.w_decade,
        w_theme=args.w_theme
    )
    
    # Save encoders
    print("Saving encoders...")
    os.makedirs(args.model_dir, exist_ok=True)
    for dim, encoder in encoders.items():
        with open(os.path.join(args.model_dir, f"{dim}_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
    
    print(f"Done! Model saved to {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train film dense retrieval model")
    parser.add_argument("--data-path", type=str, default="movies.csv",
                        help="Path to film dataset CSV")
    parser.add_argument("--base-model", type=str, default="bert-base-uncased",
                        help="Base transformer model")
    parser.add_argument("--embedding-dim", type=int, default=300,
                        help="Retrieval embedding dimension")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--model-dir", type=str, default="./output",
                        help="Output directory for model")
    parser.add_argument("--w-retr", type=float, default=1.0,
                        help="Weight for retrieval loss")
    parser.add_argument("--w-genre", type=float, default=0.1,
                        help="Weight for genre loss")
    parser.add_argument("--w-decade", type=float, default=0.1,
                        help="Weight for decade loss")
    parser.add_argument("--w-theme", type=float, default=0.1,
                        help="Weight for theme loss")
    
    args = parser.parse_args()
    main(args)