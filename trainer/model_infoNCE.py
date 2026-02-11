import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import models


class FilmRetrievalModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=300, num_genres=None, num_decades=None, num_themes=None):
        super(FilmRetrievalModel, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Main retrieval projection
        self.retrieval_projection = nn.Linear(input_dim, embedding_dim)
        
        # Classification heads for auxiliary losses
        self.genre_head = nn.Linear(embedding_dim, num_genres) if num_genres else None
        self.decade_head = nn.Linear(embedding_dim, num_decades) if num_decades else None
        self.theme_head = nn.Linear(embedding_dim, num_themes) if num_themes else None
    
    def forward(self, pooled_output, return_heads=False):
        # Project to retrieval space
        z = self.retrieval_projection(pooled_output)
        
        # L2 normalization for cosine similarity
        z_hat = F.normalize(z, p=2, dim=1)
        
        if not return_heads:
            return z_hat
        
        # Compute auxiliary head outputs
        aux_outputs = {}
        if self.genre_head is not None:
            aux_outputs['genre_logits'] = self.genre_head(z_hat)
        if self.decade_head is not None:
            aux_outputs['decade_logits'] = self.decade_head(z_hat)
        if self.theme_head is not None:
            aux_outputs['theme_logits'] = self.theme_head(z_hat)
        
        return z_hat, aux_outputs


class MultiTaskFilmLoss(nn.Module):
    def __init__(self, tau=0.05, w_retr=1.0, w_genre=0.1, w_decade=0.1, w_theme=0.1):
        super(MultiTaskFilmLoss, self).__init__()
        self.tau = tau
        self.w_retr = w_retr
        self.w_genre = w_genre
        self.w_decade = w_decade
        self.w_theme = w_theme
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def info_nce_loss(self, z_anchors, z_positives):
        batch_size = z_anchors.size(0)
        
        # Compute similarity matrix: (batch_size, batch_size)
        # sim[i,j] = z_anchor[i] · z_pos[j] / tau
        similarities = torch.mm(z_anchors, z_positives.t()) / self.tau
        
        # Labels: diagonal elements are positives (i=j)
        labels = torch.arange(batch_size, device=z_anchors.device)
        
        # InfoNCE: cross-entropy over similarity scores
        loss = F.cross_entropy(similarities, labels)
        
        return loss
    
    def forward(self, z_hat, z_pos_hat, batch):
        # Retrieval loss (InfoNCE)
        loss_retr = self.info_nce_loss(z_hat, z_pos_hat)
        total_loss = self.w_retr * loss_retr
        
        # Auxiliary classification losses
        aux_outputs = batch.get('aux_outputs', {})
        
        if 'genre_logits' in aux_outputs and 'genre_labels' in batch:
            loss_genre = self.bce_loss(aux_outputs['genre_logits'], batch['genre_labels'].float())
            total_loss = total_loss + self.w_genre * loss_genre
        
        if 'decade_logits' in aux_outputs and 'decade_labels' in batch:
            loss_decade = self.ce_loss(aux_outputs['decade_logits'], batch['decade_labels'])
            total_loss = total_loss + self.w_decade * loss_decade
        
        if 'theme_logits' in aux_outputs and 'theme_labels' in batch:
            loss_theme = self.ce_loss(aux_outputs['theme_logits'], batch['theme_labels'])
            total_loss = total_loss + self.w_theme * loss_theme
        
        return total_loss
