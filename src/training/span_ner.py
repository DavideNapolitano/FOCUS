# src/training/span_ner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class SpanNERHead(nn.Module):
    """
    Span-based Named Entity Recognition that explicitly models multi-token spans.
    
    This approach uses:
    1. BIO tagging scheme (Begin, Inside, Outside)
    2. Span representation via start/end token features
    3. Explicit boundary detection for better span identification
    
    This allows accurate recognition of entities spanning multiple tokens,
    which is essential for real-world entity names like "New York City" or
    "The White House".
    """
    def __init__(
        self,
        hidden_size: int,
        num_entity_types: int,
        max_span_length: int = 8,
        use_boundary_detection: bool = True
    ):
        super().__init__()
        
        self.num_entity_types = num_entity_types
        self.max_span_length = max_span_length
        self.use_boundary_detection = use_boundary_detection
        
        # Boundary detection - find start/end of entities
        if use_boundary_detection:
            # B: begin, I: inside, O: outside
            self.boundary_classifier = nn.Linear(hidden_size, 3)
        
        # Entity type classifier
        self.entity_classifier = nn.Linear(hidden_size * 2, num_entity_types)
        
        # Span scorer - determines if a span is a valid entity
        self.span_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
        # Span representation generator
        self.span_combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for span-based NER.
        
        Args:
            hidden_states: Language features from the model [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask to avoid including padding [batch_size, seq_length]
            
        Returns:
            Dictionary containing:
                - boundary_logits: Token boundary predictions (B/I/O) if enabled
                - span_scores: Score for each candidate span being a valid entity
                - entity_logits: Entity type predictions for each candidate span
                - candidate_spans: Start and end indices for candidate spans
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Apply attention mask if provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device).bool()
        
        outputs = {}
        
        # 1. Boundary detection (optional)
        if self.use_boundary_detection:
            boundary_logits = self.boundary_classifier(hidden_states)
            outputs['boundary_logits'] = boundary_logits
        
        # 2. Generate candidate spans
        candidate_spans = []
        valid_span_mask = []
        
        for batch_idx in range(batch_size):
            # Get valid token positions (not padding)
            valid_positions = torch.where(attention_mask[batch_idx])[0]
            
            if len(valid_positions) == 0:
                continue
                
            # Generate all possible spans within max_span_length
            batch_spans = []
            batch_valid_mask = []
            
            for start_idx in valid_positions:
                # Maximum end position is either start + max_span or sequence end
                max_end = min(start_idx + self.max_span_length, seq_length)
                
                # Generate spans from start to all possible ends
                for end_idx in range(start_idx, max_end):
                    if end_idx in valid_positions:
                        # Add candidate span (start, end) indices
                        batch_spans.append((start_idx.item(), end_idx.item()))
                        batch_valid_mask.append(1)
            
            candidate_spans.append(batch_spans)
            valid_span_mask.append(batch_valid_mask)
        
        # If no valid spans found, return empty outputs
        if not candidate_spans or all(len(spans) == 0 for spans in candidate_spans):
            outputs['span_scores'] = torch.zeros(batch_size, 0, 1, device=device)
            outputs['entity_logits'] = torch.zeros(batch_size, 0, self.num_entity_types, device=device)
            outputs['candidate_spans'] = []
            return outputs
        
        # 3. Process spans batch by batch
        all_span_scores = []
        all_entity_logits = []
        
        for batch_idx in range(batch_size):
            if not candidate_spans[batch_idx]:
                # Skip empty batches
                all_span_scores.append(torch.zeros(0, 1, device=device))
                all_entity_logits.append(torch.zeros(0, self.num_entity_types, device=device))
                continue
            
            # Extract span features
            span_features = []
            for start_idx, end_idx in candidate_spans[batch_idx]:
                # Get start and end token representations
                start_repr = hidden_states[batch_idx, start_idx]
                end_repr = hidden_states[batch_idx, end_idx]
                
                # Concatenate to form span representation
                span_repr = torch.cat([start_repr, end_repr], dim=-1)
                span_features.append(span_repr)
            
            # Stack span features
            span_features = torch.stack(span_features) if span_features else torch.zeros(0, hidden_size * 2, device=device)
            
            # Score spans
            span_scores = self.span_scorer(span_features)
            
            # Classify entity types
            entity_logits = self.entity_classifier(span_features)
            
            all_span_scores.append(span_scores)
            all_entity_logits.append(entity_logits)
        
        # Create padded tensors for batch processing
        max_spans = max(len(spans) for spans in candidate_spans)
        padded_span_scores = torch.zeros(batch_size, max_spans, 1, device=device)
        padded_entity_logits = torch.zeros(batch_size, max_spans, self.num_entity_types, device=device)
        
        for batch_idx in range(batch_size):
            if len(all_span_scores[batch_idx]) > 0:
                padded_span_scores[batch_idx, :len(all_span_scores[batch_idx])] = all_span_scores[batch_idx]
                padded_entity_logits[batch_idx, :len(all_entity_logits[batch_idx])] = all_entity_logits[batch_idx]
        
        outputs['span_scores'] = padded_span_scores
        outputs['entity_logits'] = padded_entity_logits
        outputs['candidate_spans'] = candidate_spans
        
        return outputs
    
    def compute_loss(
        self, 
        boundary_logits: Optional[torch.Tensor],
        span_scores: torch.Tensor,
        entity_logits: torch.Tensor,
        candidate_spans: List[List[Tuple[int, int]]],
        span_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss for span-based NER.
        
        Args:
            boundary_logits: Token boundary predictions (B/I/O) if boundary detection is used
            span_scores: Score for each candidate span being a valid entity
            entity_logits: Entity type predictions for each candidate span
            candidate_spans: Start and end indices for candidate spans
            span_labels: Entity type labels for each span (-100 for non-entities)
            boundary_labels: Boundary labels (B/I/O) if boundary detection is used
            
        Returns:
            Dictionary containing:
                - loss: Combined loss
                - boundary_loss: Boundary detection loss (if enabled)
                - span_loss: Span scoring loss
                - entity_loss: Entity classification loss
        """
        device = span_scores.device
        losses = {}
        
        # 1. Boundary loss (optional)
        if self.use_boundary_detection and boundary_logits is not None and boundary_labels is not None:
            active_boundary_mask = boundary_labels != -100
            
            if active_boundary_mask.sum() > 0:
                active_logits = boundary_logits.view(-1, 3)[active_boundary_mask.view(-1)]
                active_labels = boundary_labels.view(-1)[active_boundary_mask.view(-1)]
                boundary_loss = F.cross_entropy(active_logits, active_labels)
                losses['boundary_loss'] = boundary_loss
            else:
                losses['boundary_loss'] = torch.tensor(0.0, device=device)
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=device)
        
        # 2. Span scoring loss
        span_target = (span_labels != -100).float().unsqueeze(-1)
        span_loss = F.binary_cross_entropy_with_logits(
            span_scores, 
            span_target,
            reduction='mean'
        )
        losses['span_loss'] = span_loss
        
        # 3. Entity classification loss
        active_entity_mask = span_labels != -100
        
        if active_entity_mask.sum() > 0:
            active_logits = entity_logits.view(-1, self.num_entity_types)[active_entity_mask.view(-1)]
            active_labels = span_labels.view(-1)[active_entity_mask.view(-1)]
            entity_loss = F.cross_entropy(active_logits, active_labels)
            losses['entity_loss'] = entity_loss
        else:
            losses['entity_loss'] = torch.tensor(0.0, device=device)
        
        # Combine losses
        total_loss = (
            losses.get('boundary_loss', 0.0) +
            losses.get('span_loss', 0.0) +
            losses.get('entity_loss', 0.0)
        )
        losses['loss'] = total_loss
        
        return losses
        
    def decode_spans(
        self,
        span_scores: torch.Tensor,
        entity_logits: torch.Tensor,
        candidate_spans: List[List[Tuple[int, int]]],
        threshold: float = 0.5,
        tokens: Optional[List[str]] = None
    ) -> List[List[Dict[str, Union[Tuple[int, int], int, float, str]]]]:
        """
        Decode predicted spans into structured entity predictions.
        
        Args:
            span_scores: Score for each candidate span being a valid entity
            entity_logits: Entity type predictions for each candidate span
            candidate_spans: Start and end indices for candidate spans
            threshold: Confidence threshold for span detection
            tokens: Optional token list for extracting entity text
            
        Returns:
            List of lists of entity dictionaries with:
                - span: (start, end) indices
                - entity_type: Predicted entity type index
                - score: Confidence score
                - text: Entity text (if tokens provided)
        """
        batch_size = span_scores.shape[0]
        span_probs = torch.sigmoid(span_scores)
        entity_probs = F.softmax(entity_logits, dim=-1)
        
        # Extract predictions for each batch
        batch_predictions = []
        
        for batch_idx in range(batch_size):
            # Get predictions for this batch
            batch_span_probs = span_probs[batch_idx, :len(candidate_spans[batch_idx])]
            batch_entity_probs = entity_probs[batch_idx, :len(candidate_spans[batch_idx])]
            
            # Get entity type with highest probability
            max_entity_probs, max_entity_types = batch_entity_probs.max(dim=-1)
            
            # Filter by confidence threshold
            valid_spans = batch_span_probs.squeeze(-1) >= threshold
            
            # Collect valid predictions
            predictions = []
            
            for i, is_valid in enumerate(valid_spans):
                if is_valid:
                    start_idx, end_idx = candidate_spans[batch_idx][i]
                    entity_type = max_entity_types[i].item()
                    score = batch_span_probs[i].item()
                    
                    entity_dict = {
                        'span': (start_idx, end_idx),
                        'entity_type': entity_type,
                        'score': score
                    }
                    
                    # Add text if tokens provided
                    if tokens is not None:
                        entity_text = ' '.join(tokens[start_idx:end_idx+1])
                        entity_dict['text'] = entity_text
                    
                    predictions.append(entity_dict)
            
            # Sort by score
            predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Handle overlapping spans with non-maximum suppression
            final_predictions = self._non_maximum_suppression(predictions)
            
            batch_predictions.append(final_predictions)
        
        return batch_predictions
    
    def _non_maximum_suppression(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply non-maximum suppression to remove overlapping spans.
        Keep the higher scoring span when there's overlap.
        """
        if not predictions:
            return []
            
        # Sort by score
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # NMS
        kept_preds = []
        for pred in sorted_preds:
            # Check overlap with kept predictions
            should_keep = True
            
            for kept in kept_preds:
                if self._spans_overlap(pred['span'], kept['span']):
                    should_keep = False
                    break
            
            if should_keep:
                kept_preds.append(pred)
        
        return kept_preds
    
    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap"""
        start1, end1 = span1
        start2, end2 = span2
        
        # Check for overlap
        return max(start1, start2) <= min(end1, end2)