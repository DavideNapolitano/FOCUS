# src/training/multi_task_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from typing import Optional, Dict, List, Tuple, Union

from .span_ner import SpanNERHead


class PercentageBasedDetectionHead(nn.Module):
    """
    Object detection head that works with normalized percentage-based coordinates (0-1 range).
    
    This approach is ideal for vision transformers because:
    1. Vision transformers lose explicit spatial correspondence during processing
    2. Percentage coordinates are resolution-independent
    3. Coordinates remain meaningful regardless of image resizing or patch-based encoding
    """
    def __init__(self, hidden_size: int, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries
        
        # Object queries - learnable parameters that attend to different objects
        self.query_embed = nn.Embedding(num_queries, hidden_size)
        
        # Classification head - predicts class probabilities for each query
        self.class_embed = nn.Linear(hidden_size, num_classes + 1)  # +1 for background
        
        # Box prediction head - predicts normalized [x1,y1,x2,y2] coordinates
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # [x1,y1,x2,y2] in 0-1 range
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the detection head.
        
        Args:
            hidden_states: Vision features from the model [batch_size, num_patches, hidden_size]
            
        Returns:
            class_logits: Class predictions [batch_size, num_queries, num_classes+1]
            box_coords: Normalized box coordinates [batch_size, num_queries, 4]
            attention_weights: Attention weights for visualization/analysis [batch_size, num_queries, num_patches]
        """
        batch_size = hidden_states.shape[0]
        
        # Repeat query embeddings for each item in the batch
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Cross-attention: queries attend to image features
        attention = torch.bmm(queries, hidden_states.transpose(1, 2))
        attention_weights = F.softmax(attention, dim=-1)
        
        # Aggregate features for each query based on attention
        attended_features = torch.bmm(attention_weights, hidden_states)
        
        # Predict class probabilities
        class_logits = self.class_embed(attended_features)
        
        # Predict normalized box coordinates (0-1 range)
        # Using sigmoid ensures outputs are in the 0-1 range
        box_coords = self.bbox_embed(attended_features).sigmoid()
        
        return class_logits, box_coords, attention_weights


class MultiTaskQwen2VL(nn.Module):
    """
    Multi-task wrapper for Qwen2-VL model that adds object detection and span-based named entity recognition.
    
    This approach uses auxiliary tasks to enhance the vision-language model's understanding by:
    1. Forcing explicit object detection improves visual grounding
    2. Adding span-based named entity recognition improves entity comprehension in text
    3. Optional contrastive learning aligns visual objects with textual entities
    
    The percentage-based coordinate system ensures compatibility with the vision transformer
    architecture and provides resolution independence.
    """
    def __init__(
        self, 
        base_model: Union[Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration], 
        object_detection_layer: int = 11, 
        ner_layer: int = 11,
        vision_layer_percentage: float = 0.1,  # Use 0.5 for 50%
        language_layer_percentage: float = 0.1,
        num_object_classes: int = 80, 
        num_entity_types: int = 10,
        max_span_length: int = 8,
        det_loss_weight: float = 0.5, 
        ner_loss_weight: float = 0.5, 
        contrastive_loss_weight: float = 0.2
    ):
        super().__init__()
        self.base_model = base_model
        
        # Extract hidden size from the base model
        hidden_size = self.base_model.config.hidden_size
        
        # Create auxiliary task heads
        self.detection_head = PercentageBasedDetectionHead(hidden_size, num_object_classes)
        
        # Use span-based NER instead of token classification
        self.ner_head = SpanNERHead(
            hidden_size=hidden_size,
            num_entity_types=num_entity_types,
            max_span_length=max_span_length,
            use_boundary_detection=True
        )
        
        # # Layer indices to extract features from (middle-to-upper layers work best)
        # self.object_detection_layer = object_detection_layer
        # self.ner_layer = ner_layer

        # Calculate actual layer indices based on percentages
        vision_layers = len(base_model.visual.layers)
        language_layers = len(base_model.model.layers)
        
        self.object_detection_layer = int(vision_layers * vision_layer_percentage)
        self.ner_layer = int(language_layers * language_layer_percentage)

        print(f"Using vision layer {self.object_detection_layer}/{vision_layers} ({vision_layer_percentage*100}%)")
        print(f"Using language layer {self.ner_layer}/{language_layers} ({language_layer_percentage*100}%)")
        
        # Loss weights for multi-task learning
        self.det_loss_weight = det_loss_weight
        self.ner_loss_weight = ner_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        
        # Track features for feature extraction
        self.vision_features = None
        self.text_features = None
        
        # Register hooks to extract intermediate features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations"""
        
        def vision_hook_fn(module, input, output):
            self.vision_features = output
            
        def text_hook_fn(module, input, output):
            self.text_features = output
            
        # For vision tower
        if hasattr(self.base_model, 'visual') and self.base_model.visual is not None:
            if hasattr(self.base_model.visual, 'layers'):
                target_vision_layer = self.base_model.visual.layers[self.object_detection_layer]
                target_vision_layer.register_forward_hook(vision_hook_fn)
        
        # For language model
        if hasattr(self.base_model, 'model') and self.base_model.model is not None:
            if hasattr(self.base_model.model, 'layers'):
                target_text_layer = self.base_model.model.layers[self.ner_layer]
                target_text_layer.register_forward_hook(text_hook_fn)
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        pixel_values=None, 
        pixel_values_videos=None, 
        image_grid_thw=None, 
        video_grid_thw=None, 
        labels=None, 
        detection_targets=None, 
        ner_targets=None, 
        ner_boundary_labels=None,
        span_labels=None,
        **kwargs
    ):
        """
        Forward pass with multi-task learning.
        
        This method:
        1. Runs the base vision-language model
        2. Extracts intermediate features via hooks
        3. Computes auxiliary task predictions and losses
        4. Combines losses with appropriate weighting
        
        The training objective becomes a weighted sum of:
        - Language modeling loss (base model)
        - Object detection loss (auxiliary)
        - Named entity recognition loss (auxiliary)
        - Optional contrastive loss (cross-modal alignment)
        """
        # Reset feature storage
        self.vision_features = None
        self.text_features = None
        
        # Forward pass through base model (this will trigger hooks)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            **kwargs
        )
        
        # Get base model loss
        base_loss = outputs.loss if outputs.loss is not None else 0.0
        combined_loss = base_loss
        
        # Add auxiliary losses
        auxiliary_outputs = {}
        
        # 1. Object detection loss
        det_loss = None
        if self.vision_features is not None and detection_targets is not None:
            pred_classes, pred_boxes, attn_weights = self.detection_head(self.vision_features)
            det_loss = self._compute_detection_loss(
                pred_classes, pred_boxes, detection_targets)
            combined_loss += self.det_loss_weight * det_loss
            auxiliary_outputs['detection_loss'] = det_loss
            auxiliary_outputs['detection_predictions'] = {
                'classes': pred_classes,
                'boxes': pred_boxes
            }
        
        # 2. NER loss - now using span-based NER
        ner_loss = None
        if self.text_features is not None:
            # Get NER predictions
            ner_outputs = self.ner_head(
                self.text_features,
                attention_mask=attention_mask
            )
            
            # Store NER outputs
            auxiliary_outputs['ner_outputs'] = ner_outputs
            
            # Compute NER loss if targets provided
            if span_labels is not None:
                ner_losses = self.ner_head.compute_loss(
                    boundary_logits=ner_outputs.get('boundary_logits'),
                    span_scores=ner_outputs.get('span_scores'),
                    entity_logits=ner_outputs.get('entity_logits'),
                    candidate_spans=ner_outputs.get('candidate_spans'),
                    span_labels=span_labels,
                    boundary_labels=ner_boundary_labels
                )
                
                ner_loss = ner_losses['loss']
                combined_loss += self.ner_loss_weight * ner_loss
                auxiliary_outputs['ner_loss'] = ner_loss
                
                # Store detailed NER losses
                for loss_name, loss_value in ner_losses.items():
                    if loss_name != 'loss':
                        auxiliary_outputs[f'ner_{loss_name}'] = loss_value
        
        # 3. Optional contrastive loss
        contrastive_loss = None
        if self.contrastive_loss_weight > 0 and self.vision_features is not None and self.text_features is not None:
            contrastive_loss = self._compute_contrastive_loss(
                self.vision_features, self.text_features)
            combined_loss += self.contrastive_loss_weight * contrastive_loss
            auxiliary_outputs['contrastive_loss'] = contrastive_loss
        
        # Update the loss in outputs
        outputs.loss = combined_loss
        
        # Add auxiliary outputs to the returned object
        for key, value in auxiliary_outputs.items():
            setattr(outputs, key, value)
        
        return outputs
    
    def _compute_detection_loss(self, pred_classes, pred_boxes, targets):
        """
        Compute detection loss using percentage-based coordinates.
        
        This loss consists of:
        1. Classification loss (cross-entropy)
        2. Box regression loss (L1 loss on normalized coordinates)
        
        Using percentage-based coordinates eliminates the need to scale
        the box regression loss based on image size.
        """
        batch_size, num_queries = pred_classes.shape[:2]
        
        # Classification loss (filter by target presence)
        target_classes = torch.zeros(
            (batch_size, num_queries), 
            dtype=torch.long, 
            device=pred_classes.device
        )
        
        # For simplicity, assign targets to predictions by order
        # (A proper implementation would use Hungarian matching)
        num_target_boxes = min(num_queries, targets['boxes'].size(1))
        
        if num_target_boxes > 0:
            # Assign 1-indexed class labels to the first num_target_boxes queries
            target_classes[:, :num_target_boxes] = targets['classes'][:, :num_target_boxes] + 1
            
            # Class loss - cross entropy for all queries
            class_loss = F.cross_entropy(
                pred_classes.reshape(-1, pred_classes.size(-1)), 
                target_classes.reshape(-1)
            )
            
            # Box loss - L1 loss only for positive examples (non-background)
            box_loss = F.l1_loss(
                pred_boxes[:, :num_target_boxes], 
                targets['boxes'][:, :num_target_boxes], 
                reduction='none'
            )
            
            # Sum over box dimensions, mean over batch and queries
            box_loss = box_loss.sum(-1).mean()
            
            return class_loss + box_loss
        else:
            # Only classification loss when no boxes
            return F.cross_entropy(
                pred_classes.reshape(-1, pred_classes.size(-1)), 
                target_classes.reshape(-1)
            )
    
    def _compute_contrastive_loss(self, vision_features, text_features):
        """
        Compute contrastive loss to align visual and textual representations.
        
        This helps the model learn correspondences between:
        1. Objects detected in images
        2. Entities mentioned in text
        
        The temperature parameter controls the sharpness of the distribution.
        """
        # Global pooling to get compact representations
        vision_emb = vision_features.mean(dim=1)  # [batch_size, hidden_size]
        text_emb = text_features.mean(dim=1)      # [batch_size, hidden_size]
        
        # L2 normalization
        vision_emb = F.normalize(vision_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.matmul(vision_emb, text_emb.transpose(0, 1))
        
        # Temperature parameter (smaller = sharper distribution)
        temperature = 0.07
        similarity = similarity / temperature
        
        # InfoNCE loss
        # In-batch negatives: diagonal elements are positive pairs (same index)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        loss = F.cross_entropy(similarity, labels)
        
        return loss