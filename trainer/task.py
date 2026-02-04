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

def create_hierarchical_model(base_model="bert-base-uncased", embedding_dim=256):
    model = HierarchicalFilmEmbedding(base_model, embedding_dim)
    
    # Create a sentence transformer wrapper
    word_embedding_model = models.Transformer(base_model, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    
    # This is a base ST model that we'll replace the forward method on
    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Replace the encode method to use our hierarchical model
    original_encode = st_model.encode
    
    def hierarchical_encode(sentences, *args, **kwargs):
        # Keep the original encode functionality for tokenization
        features = original_encode(sentences, *args, convert_to_tensor=True, **kwargs)
        # Apply our hierarchical model
        return model(features)
    
    st_model.encode = hierarchical_encode
    
    return st_model

def train_hierarchical_model(train_dataset, val_dataset, model, output_path, epochs=10):
    # Define our custom loss
    train_loss = MultiDimensionTripletLoss(model=model, margin=0.5)
    
    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    
    # Create evaluators for each dimension
    evaluators = {}
    dimensions = ['genre', 'theme', 'era', 'director', 'joint']
    
    for dim in dimensions:
        # Create evaluation pairs for this dimension
        eval_examples = create_evaluation_pairs_for_dimension(val_dataset, dim)
        
        evaluators[dim] = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples,
            name=f'{dim}-validation',
            batch_size=32
        )
    
    # Train with multiple evaluators
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        evaluator=evaluators,
        evaluation_steps=1000,
        warmup_steps=100,
        output_path=output_path
    )
    
    return model

def extract_dimension_embeddings(model, texts, dimension):
    """Extract embeddings for a specific dimension"""
    all_embeddings = model.encode(texts, batch_size=32)
    # Extract only the embeddings for the requested dimension
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
    model = create_hierarchical_model(args.base_model, args.embedding_dim)
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Train model
    print("Training model...")
    model = train_hierarchical_model(
        train_dataset, 
        val_dataset, 
        model, 
        output_path=args.model_dir,
        epochs=args.epochs
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