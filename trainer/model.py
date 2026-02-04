import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import models

class HierarchicalFilmEmbedding(nn.Module):
    def __init__(self, base_model_name, embedding_dim=256):
        super(HierarchicalFilmEmbedding, self).__init__()
        # Base text encoder (shared across all dimensions)
        self.text_encoder = models.Transformer(base_model_name, max_seq_length=512)
        self.pooling = models.Pooling(self.text_encoder.get_word_embedding_dimension(), pooling_mode='mean')
        
        # Specialized embedding projections for different aspects
        self.genre_projection = nn.Sequential(
            nn.Linear(self.pooling.get_sentence_embedding_dimension(), 512),
            nn.Tanh(),
            nn.Linear(512, embedding_dim)
        )
        
        self.theme_projection = nn.Sequential(
            nn.Linear(self.pooling.get_sentence_embedding_dimension(), 512),
            nn.Tanh(),
            nn.Linear(512, embedding_dim)
        )
        
        self.era_projection = nn.Sequential(
            nn.Linear(self.pooling.get_sentence_embedding_dimension(), 512),
            nn.Tanh(),
            nn.Linear(512, embedding_dim)
        )
        
        self.director_projection = nn.Sequential(
            nn.Linear(self.pooling.get_sentence_embedding_dimension(), 512),
            nn.Tanh(),
            nn.Linear(512, embedding_dim)
        )
        
        # Final joint embedding (combines all aspects)
        self.joint_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, 512),
            nn.Tanh(),
            nn.Linear(512, embedding_dim)
        )
        
    def forward(self, features):
        # Get base text representation
        token_embeddings = self.text_encoder(features)['token_embeddings']
        pooled_output = self.pooling({'token_embeddings': token_embeddings})['sentence_embedding']
        
        # Get specialized embeddings
        genre_embedding = self.genre_projection(pooled_output)
        theme_embedding = self.theme_projection(pooled_output)
        era_embedding = self.era_projection(pooled_output)
        director_embedding = self.director_projection(pooled_output)
        
        # Combine for joint embedding
        combined = torch.cat([
            genre_embedding, 
            theme_embedding, 
            era_embedding, 
            director_embedding
        ], dim=1)
        
        joint_embedding = self.joint_projection(combined)
        
        return {
            'genre': genre_embedding,
            'theme': theme_embedding,
            'era': era_embedding,
            'director': director_embedding,
            'joint': joint_embedding
        }

class MultiDimensionTripletLoss(nn.Module):
    def __init__(self, model, margin=0.5):
        super(MultiDimensionTripletLoss, self).__init__()
        self.model = model
        self.margin = margin
        
    def forward(self, sentence_features, labels):
        # Extract anchor, positive and negative texts
        reps = self.model.encode(sentence_features, convert_to_tensor=True)
        
        dimensions = labels['dimension']  # Which dimension this triplet belongs to
        loss = 0
        
        # Process each triplet according to its dimension
        for i in range(0, len(reps), 3):
            anchor, positive, negative = reps[i], reps[i+1], reps[i+2]
            dim = dimensions[i//3]
            
            # Get the specific embeddings for this dimension
            anchor_emb = anchor[dim]
            positive_emb = positive[dim]
            negative_emb = negative[dim]
            
            # Calculate triplet loss for this dimension
            distance_pos = F.pairwise_distance(anchor_emb, positive_emb)
            distance_neg = F.pairwise_distance(anchor_emb, negative_emb)
            
            # Apply margin
            triplet_loss = F.relu(distance_pos - distance_neg + self.margin)
            
            # Add to total loss
            loss += triplet_loss.mean()
            
            # Also add joint embedding loss with a smaller weight
            anchor_joint = anchor['joint']
            positive_joint = positive['joint']
            negative_joint = negative['joint']
            
            distance_pos_joint = F.pairwise_distance(anchor_joint, positive_joint)
            distance_neg_joint = F.pairwise_distance(anchor_joint, negative_joint)
            
            joint_loss = F.relu(distance_pos_joint - distance_neg_joint + self.margin)
            loss += 0.25 * joint_loss.mean()  # Lower weight for joint loss
            
        return loss / (len(reps) // 3)  # Average loss per triplet