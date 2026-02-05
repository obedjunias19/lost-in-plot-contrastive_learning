import os
import argparse
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, evaluation
from sklearn.neighbors import KNeighborsClassifier

# Import our modules
from trainer.model import HierarchicalFilmEmbedding, MultiDimensionTripletLoss
from trainer.dataset import MultiTaskFilmDataset, create_evaluation_pairs_for_dimension

def create_hierarchical_model(base_model="bert-base-uncased", embedding_dim=256, device=None):
    # Create transformer + pooling that produce pooled sentence embeddings
    word_embedding_model = models.Transformer(base_model, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

    input_dim = pooling_model.get_sentence_embedding_dimension()

    # Projection heads (trainable)
    hier_proj = HierarchicalFilmEmbedding(input_dim=input_dim, embedding_dim=embedding_dim)

    # Base SentenceTransformer for tokenization + pooling (we won't fine-tune transformer by default)
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Register the projection module so it's part of the nn.Module tree
    st_model.add_module('hier_proj', hier_proj)

    # Keep original encode that returns pooled embeddings
    original_encode = st_model.encode

    def hierarchical_encode(sentences, convert_to_tensor=True, batch_size=32, **kwargs):
        # Get pooled embeddings from the underlying transformer+pooling
        pooled = original_encode(sentences, convert_to_tensor=True, batch_size=batch_size, **kwargs)

        # Ensure tensor on correct device
        if device is not None:
            pooled = pooled.to(device)

        # Apply projection heads -> dict of tensors (batch, emb_dim)
        out = st_model.hier_proj(pooled)

        # Convert to list of per-sample dicts for downstream loss/eval
        batch_size_actual = out['joint'].size(0)
        results = []
        for i in range(batch_size_actual):
            entry = {k: out[k][i] for k in out}
            results.append(entry)

        return results

    st_model.encode = hierarchical_encode

    return st_model

def train_hierarchical_model(train_dataset, val_dataset, st_model, output_path, epochs=10, batch_size=32, lr=2e-5, device=None):
    # Simple PyTorch training loop that updates the projection heads registered
    # on the SentenceTransformer `st_model` (module name: 'hier_proj').

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    # Loss wrapper uses st_model.encode to obtain per-sample dicts
    train_loss_fn = MultiDimensionTripletLoss(st_model, margin=0.5, device=device)

    # Only optimize the projection heads by default
    params = list(st_model.hier_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    st_model.to(device)
    st_model.hier_proj.to(device)

    for epoch in range(epochs):
        st_model.train()
        total_loss = 0.0
        steps = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            loss_val = train_loss_fn(batch)
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{epochs} — train loss: {avg_loss:.4f}")

        # Optional simple validation: compute average loss on a small subset
        st_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            vsteps = 0
            for vb in val_dataloader:
                l = train_loss_fn(vb)
                val_loss += l.item()
                vsteps += 1
                if vsteps >= 20:
                    break
            if vsteps > 0:
                print(f"  Validation loss (sampled): {val_loss / vsteps:.4f}")

    return st_model

def extract_dimension_embeddings(model, texts, dimension):
    """Extract embeddings for a specific dimension"""
    all_embeddings = model.encode(texts, batch_size=32, convert_to_tensor=True)
    # all_embeddings is a list of dicts; extract requested dimension
    dimension_embeddings = np.array([emb[dimension].cpu().numpy() for emb in all_embeddings])
    return dimension_embeddings

def load_data(data_path):
    """Load and preprocess the film dataset"""
    df = pd.read_csv(data_path)
    
    # Create label encoders for each dimension
    encoders = {}
    dimensions = ['genre', 'theme', 'era', 'director']
    
    for dim in dimensions:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.fit(df[dim].fillna('unknown'))
        encoders[dim] = encoder
    
    # Split into train/val
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    return train_df, val_df, encoders

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_df, val_df, encoders = load_data(args.data_path)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MultiTaskFilmDataset(train_df, encoders)
    val_dataset = MultiTaskFilmDataset(val_df, encoders)
    
    # Create hierarchical model
    print("Creating model...")
    model = create_hierarchical_model(args.base_model, args.embedding_dim, device=device)
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Train model
    print("Training model...")
    model = train_hierarchical_model(
        train_dataset,
        val_dataset,
        model,
        output_path=args.model_dir,
        epochs=args.epochs,
        device=device
    )
    
    # Create specialized KNN classifiers for each dimension
    print("Creating classifiers...")
    classifiers = {}
    dimensions = ['genre', 'theme', 'era', 'director']
    
    for dim in dimensions:
        # Encode texts for this dimension
        train_embeddings = extract_dimension_embeddings(model, train_df["combined_text_baseline"].tolist(), dim)
        
        # Get labels
        train_labels = encoders[dim].transform(train_df[dim])
        
        # Train classifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(train_embeddings, train_labels)
        classifiers[dim] = knn
    
    # Save everything
    print("Saving models and classifiers...")
    
    # Save each classifier
    for dim, classifier in classifiers.items():
        with open(os.path.join(args.model_dir, f"{dim}_classifier.pkl"), "wb") as f:
            pickle.dump(classifier, f)
    
    # Save each encoder
    for dim, encoder in encoders.items():
        with open(os.path.join(args.model_dir, f"{dim}_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
    
    print(f"Training complete. Model saved to {args.model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hierarchical film embedding model")
    parser.add_argument("--data-path", type=str, default="gs://your-bucket/film_dataset.csv",
                        help="Path to the film dataset CSV")
    parser.add_argument("--base-model", type=str, default="bert-base-uncased",
                        help="Base transformer model name")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Dimension of the specialized embeddings")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("AIP_MODEL_DIR", "/tmp/output"),
                        help="Directory to save model artifacts")
    
    args = parser.parse_args()
    main(args)