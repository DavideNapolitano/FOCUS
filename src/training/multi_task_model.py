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
        vision_layer_percentage: float = 0.1,  # Use 0.1 for 10% (early layer)
        language_layer_percentage: float = 0.1, # Use 0.1 for 10% (early layer)
        num_object_classes: int = 80, 
        num_entity_types: int = 10,
        max_span_length: int = 8,
        det_loss_weight: float = 0.5, 
        ner_loss_weight: float = 0.5, 
        contrastive_loss_weight: float = 0.2
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        
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
        
        # Get number of layers in model
        language_layers = len(base_model.model.layers)
        
        # For Qwen models, vision features are processed through the same language model stack
        # So we'll use the same number of layers for both
        vision_layers = language_layers
        
        # Calculate actual layer indices based on percentages
        self.object_detection_layer = max(0, min(vision_layers - 1, 
                                               int(vision_layers * vision_layer_percentage)))
        self.ner_layer = max(0, min(language_layers - 1, 
                                  int(language_layers * language_layer_percentage)))
        
        print(f"Using vision features from layer {self.object_detection_layer}/{vision_layers} "
              f"({vision_layer_percentage*100:.1f}%)")
        print(f"Using language features from layer {self.ner_layer}/{language_layers} "
              f"({language_layer_percentage*100:.1f}%)")
        
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
        """
        Register forward hooks to capture intermediate activations.
        
        This is critical for extracting features from specific layers of the model.
        The hook functions properly handle outputs whether they are tuples or tensors.
        """
        def vision_hook_fn(module, input, output):
            # Handle outputs that may be tuples (common in transformers)
            if isinstance(output, tuple):
                self.vision_features = output[0]  # Usually the first element is the hidden states
            else:
                self.vision_features = output
            
        def text_hook_fn(module, input, output):
            # Handle outputs that may be tuples (common in transformers)
            if isinstance(output, tuple):
                self.text_features = output[0]  # Usually the first element is the hidden states
            else:
                self.text_features = output
        
        # For Qwen2.5-VL and similar models, both vision and text features 
        # are processed through the same decoder stack
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Register hook for object detection
            if self.object_detection_layer < len(self.base_model.model.layers):
                target_vision_layer = self.base_model.model.layers[self.object_detection_layer]
                target_vision_layer.register_forward_hook(vision_hook_fn)
                print(f"Registered vision hook on language model layer {self.object_detection_layer}")
            
            # Register hook for NER
            if self.ner_layer < len(self.base_model.model.layers):
                target_text_layer = self.base_model.model.layers[self.ner_layer]
                target_text_layer.register_forward_hook(text_hook_fn)
                print(f"Registered text hook on language model layer {self.ner_layer}")
        else:
            print("WARNING: Could not register hooks - model structure not as expected")
            print(f"Model has attribute 'model': {hasattr(self.base_model, 'model')}")
            if hasattr(self.base_model, 'model'):
                print(f"Model has attribute 'layers': {hasattr(self.base_model.model, 'layers')}")
    
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
        """
        # Reset feature storage
        self.vision_features = None
        self.text_features = None
        
        # Create kwargs dict with only supported parameters for base model
        base_model_kwargs = {}
        
        # Add explicitly supported parameters only if they're not None
        supported_params = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'pixel_values_videos': pixel_values_videos,
            'image_grid_thw': image_grid_thw,
            'video_grid_thw': video_grid_thw,
            'labels': labels,
        }
        
        for param_name, param_value in supported_params.items():
            if param_value is not None:
                base_model_kwargs[param_name] = param_value
        
        # Parameters we know should be filtered out:
        # 1. Our auxiliary task parameters
        # 2. Trainer-specific parameters
        filtered_params = [
            # Auxiliary task parameters
            'detection_targets', 'ner_targets', 'ner_boundary_labels',
            'span_labels', 'candidate_spans',
            
            # Trainer-specific parameters
            'num_items_in_batch', 'return_loss',
            
            # Any other parameters that might cause conflicts
            'second_per_grid_ts'  # Sometimes passed by Qwen2.5 processor
        ]
        
        # Add any remaining kwargs that aren't in the filtered list
        for k, v in kwargs.items():
            if k not in filtered_params:
                base_model_kwargs[k] = v
        
        # Print filtered parameters for debugging (only during first forward pass)
        if not hasattr(self, '_printed_filtered_params'):
            filtered_keys = [k for k in kwargs.keys() if k in filtered_params]
            if filtered_keys:
                print(f"Filtering out parameters: {filtered_keys}")
            self._printed_filtered_params = True
        
        # Add this right before base_model call in the forward method
        # # Set requires_grad=True on input_ids tensor if gradient checkpointing is enabled
        # if hasattr(self.base_model, 'gradient_checkpointing') and self.base_model.gradient_checkpointing:
        #     if 'input_ids' in base_model_kwargs and base_model_kwargs['input_ids'] is not None:
        #         if base_model_kwargs['input_ids'].requires_grad == False:
        #             base_model_kwargs['input_ids'].requires_grad_(True)
        #             print("Enabled requires_grad on input_ids")

        # print(f"Base model kwargs: {base_model_kwargs}")

        # # set basel_mode_kwargs requires_grad=True
        # for k, v in base_model_kwargs.items():
        #     # if v is a tensor 
        #     if isinstance(v, torch.Tensor):
        #         # print requires_grad status
        #         print(f"Base model kwargs {k}({v.dtype}) requires_grad: {v.requires_grad}")
        #         # set requires_grad=True
        #         if v.requires_grad == False:
        #             v.requires_grad_(True)
        #             print(f"Enabled requires_grad on base model kwargs {k}")

        # Forward pass through base model with filtered arguments
        outputs = self.base_model(**base_model_kwargs)
        
        # Get base model loss
        base_loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else 0.0
        combined_loss = base_loss
        
        # Add auxiliary losses
        auxiliary_outputs = {}
        
        # print(f"Vision features: {self.vision_features}")
        # print(f"Text features: {self.text_features}")

        # set self.vision_features requires_grad=True
        if self.vision_features is not None:
            self.vision_features.requires_grad_(True)
            # print("Enabled requires_grad on vision features")
        if self.text_features is not None:
            self.text_features.requires_grad_(True)
            # print("Enabled requires_grad on text features")

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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            print("Gradient checkpointing enabled for base model")
        else:
            print("Warning: base model does not support gradient_checkpointing_enable")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
            print("Gradient checkpointing disabled for base model")
        else:
            print("Warning: base model does not support gradient_checkpointing_disable")
    
    def _set_gradient_checkpointing(self, module, value=False):
        """Set gradient checkpointing for the base model"""
        if hasattr(self.base_model, '_set_gradient_checkpointing'):
            self.base_model._set_gradient_checkpointing(module, value)
        else:
            print("Warning: base model does not support _set_gradient_checkpointing")
    
    def enable_input_require_grads(self):
        """
        Enable input gradients for gradient checkpointing in a way compatible with 
        integer input tensors.
        """
        # This is the correct way to handle for transformer models with integer inputs
        def make_inputs_require_grad(module, input, output):
            # Output of the embedding layer is a float tensor and CAN have requires_grad=True
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                output[0].requires_grad_(True)
        
        # Find and register hook on embedding layer (not the inputs)
        if hasattr(self.base_model, 'get_input_embeddings'):
            self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print("Registered gradient-enabling hook on embedding layer outputs")
        
        # For good measure, also ensure embedding weights have requires_grad=True
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            if not self.base_model.model.embed_tokens.weight.requires_grad:
                self.base_model.model.embed_tokens.weight.requires_grad_(True)
                print("Set requires_grad=True on embedding weights")