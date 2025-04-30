# src/training/multi_task_data.py
import os
import copy
import json
import torch
import transformers
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info

from .data import SupervisedDataset, DataCollatorForSupervisedDataset
from .constants import *


class MultiTaskDataset(SupervisedDataset):
    """
    Dataset for multi-task supervised fine-tuning with auxiliary tasks.
    
    This dataset extends the SupervisedDataset to handle:
    1. Object detection annotations (COCO format with percentage-based coordinates)
    2. Span-based named entity recognition annotations
    3. Integration with the base VLM training data
    
    Using percentage-based coordinates ensures compatibility across different
    image sizes and resolutions, which is essential for vision transformers.
    """
    
    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args,
        model_id,
        padding=True,
    ):
        # Initialize base dataset
        super(MultiTaskDataset, self).__init__(
            data_path=data_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id,
            padding=padding
        )
        
        # Add parameters for auxiliary tasks
        self.include_detection = data_args.include_detection
        self.include_ner = data_args.include_ner
        self.detection_annotation_path = data_args.detection_annotation_path
        self.ner_annotation_path = data_args.ner_annotation_path
        self.max_span_length = getattr(data_args, 'max_span_length', 8)
        
        # For span-based NER
        self.use_boundary_labels = getattr(data_args, 'use_boundary_labels', True)
        self.num_entity_types = data_args.num_entity_types
        
        # Load annotations if needed
        self.detection_annotations = {}
        self.ner_annotations = {}
        
        if self.include_detection and self.detection_annotation_path:
            self._load_detection_annotations()
            
        if self.include_ner and self.ner_annotation_path:
            self._load_ner_annotations()
    
    def _load_detection_annotations(self):
        """
        Load object detection annotations with percentage-based coordinates.
        
        Expected format is COCO-style, with:
        - annotations: list of objects with bbox, image_id, category_id
        - images: list of image info with id, width, height
        - categories: list of category info with id, name
        
        Converts all coordinates to normalized [0,1] range regardless of original image size.
        """
        print(f"Loading detection annotations from {self.detection_annotation_path}")
        try:
            with open(self.detection_annotation_path, 'r') as f:
                annotations = json.load(f)
            print(f"Loaded {len(annotations.get('annotations', []))} detection annotations")
            
            # Create image info lookup for dimensions
            image_info = {}
            if 'images' in annotations:
                for img in annotations['images']:
                    image_info[img['id']] = {
                        'width': img['width'],
                        'height': img['height'],
                        'file_name': img['file_name']
                    }
                    
                    # Also store by filename for easier lookup
                    image_info[os.path.basename(img['file_name'])] = {
                        'id': img['id'],
                        'width': img['width'],
                        'height': img['height']
                    }

            print(f"Loaded {len(image_info)} images")
            
            # Build category map
            category_map = {}
            if 'categories' in annotations:
                for cat in annotations['categories']:
                    category_map[cat['id']] = {
                        'id': cat['id'],
                        'name': cat['name']
                    }

            print(f"Loaded {len(category_map)} categories")
            
            # Organize annotations by image ID and filename
            if 'annotations' in annotations:
                for ann in annotations['annotations']:
                    img_id = ann['image_id']
                    
                    # Get image filename from ID
                    img_filename = None
                    if img_id in image_info:
                        img_filename = os.path.basename(image_info[img_id]['file_name'])
                    
                    # print(f"Processing annotation for image ID: {img_id}, filename: {img_filename}")
                    
                    # Store by both ID and filename for flexibility
                    for key in [img_id, img_filename]:
                        if key is None:
                            continue
                            
                        if key not in self.detection_annotations:
                            self.detection_annotations[key] = {
                                'boxes': [],
                                'classes': [],
                                'img_info': image_info.get(img_id, {})
                            }
                        
                        # Get original image dimensions
                        img_w = image_info[img_id]['width']
                        img_h = image_info[img_id]['height']
                        
                        # Normalize coordinates to [0,1] range - PERCENTAGE BASED
                        bbox = ann['bbox']  # [x, y, width, height] in COCO format
                        x1 = bbox[0] / img_w
                        y1 = bbox[1] / img_h
                        x2 = (bbox[0] + bbox[2]) / img_w  # Convert to [x1,y1,x2,y2] format
                        y2 = (bbox[1] + bbox[3]) / img_h
                        
                        # Clip to ensure values are in [0,1]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(1, x2), min(1, y2)
                        
                        normalized_bbox = [x1, y1, x2, y2]
                        
                        self.detection_annotations[key]['boxes'].append(normalized_bbox)
                        self.detection_annotations[key]['classes'].append(ann['category_id'])
                
            print(f"Loaded annotations for {len(self.detection_annotations)} images")
        except Exception as e:
            print(f"Error loading detection annotations: {e}")
            # Continue without detection annotations
            self.include_detection = False
    
    def _load_ner_annotations(self):
        """
        Load named entity recognition annotations for span-based NER.
        
        Expected format: JSON with:
        {
            "conversation_id": {
                "spans": [[start_idx, end_idx, entity_type], ...],
                "entities": ["entity1", "entity2", ...]
            },
            ...
        }
        
        This format explicitly supports multi-token spans (e.g., "New York").
        """
        print(f"Loading NER annotations from {self.ner_annotation_path}")
        try:
            with open(self.ner_annotation_path, 'r') as f:
                self.ner_annotations = json.load(f)
            print(f"Loaded NER annotations for {len(self.ner_annotations)} conversations")
        except Exception as e:
            print(f"Error loading NER annotations: {e}")
            # Continue without NER annotations
            self.include_ner = False
    
    def get_image_id(self, source):
        """Extract image ID from the source data"""
        # Try standard ID field
        image_id = source.get('id', None)
        # print(f"Extracting image ID: {image_id}")
        image_id = int(str(image_id)[:-3])
        # print(f"Extracted image ID: {image_id}")
        
        # # If no ID, try using image filename
        # if image_id is None and 'image' in source:
        #     image_file = source['image']
        #     if isinstance(image_file, list):
        #         image_file = image_file[0]  # Use first image in multi-image case
        #     image_id = os.path.basename(image_file)
        
        return image_id
    
    def _prepare_bio_labels(self, input_ids, spans):
        """
        Create BIO (Begin, Inside, Outside) labels for tokens.
        This is used for boundary detection in span-based NER.
        
        Args:
            input_ids: Token IDs from tokenizer
            spans: List of entity spans [start, end, entity_type]
            
        Returns:
            Tensor of BIO labels for each token
                0: O (Outside)
                1: B (Begin)
                2: I (Inside)
        """
        # Initialize all tokens as outside (O) - BIO scheme
        bio_labels = torch.zeros_like(input_ids)
        
        # Fill in spans with B (begin) and I (inside) tags
        for start, end, _, _, _, _ in spans:
            if start < bio_labels.size(0) and end <= bio_labels.size(0):
                # Mark begin token
                bio_labels[start] = 1  # B
                
                # Mark inside tokens
                if end > start + 1:
                    bio_labels[start+1:end] = 2  # I
        
        return bio_labels
    
    def _prepare_span_labels(self, input_ids, spans, max_span_length):
        """
        Create span labels for span-based NER.
        
        Args:
            input_ids: Token IDs from tokenizer
            spans: List of entity spans [start, end, entity_type]
            max_span_length: Maximum span length to consider
            
        Returns:
            span_labels: Tensor of entity type labels for all possible spans (-100 for non-entities)
            candidate_spans: List of all candidate spans [(start, end), ...]
        """
        seq_length = input_ids.size(0)
        
        # Generate all possible spans up to max_span_length
        candidate_spans = []
        for start in range(seq_length):
            for end in range(start, min(start + max_span_length, seq_length)):
                candidate_spans.append((start, end))
        
        # Initialize all spans as non-entities (-100)
        span_labels = torch.full((len(candidate_spans),), -100, dtype=torch.long)
        
        # Fill in entity spans with their types
        for i, (start, end) in enumerate(candidate_spans):
            for s_start, s_end, entity_type in spans:
                # Check if this span matches an entity span
                if start == s_start and end == s_end:
                    span_labels[i] = entity_type
                    break
        
        return span_labels, candidate_spans
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Get dataset item with auxiliary task targets.
        
        Extends the base __getitem__ by adding:
        1. Detection targets (percentage-based boxes and classes)
        2. Span-based NER targets (span labels and boundary labels)
        """
        # Get base data from parent class
        data_dict = super().__getitem__(i)
        source = self.list_data_dict[i]
        
        # Add detection targets if available
        if self.include_detection:
            image_id = self.get_image_id(source)
            
            if image_id in self.detection_annotations:
                ann = self.detection_annotations[image_id]
                
                if ann['boxes'] and ann['classes']:
                    # Convert to tensors with appropriate shapes
                    boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
                    classes = torch.tensor(ann['classes'], dtype=torch.long)
                    
                    # Store targets in the expected format
                    data_dict['detection_targets'] = {
                        'boxes': boxes,
                        'classes': classes
                    }
        
        # Add NER targets if available
        if self.include_ner:
            conversation_id = source.get('id', None)
            tmp=f"conversation_{conversation_id}"
            
            if tmp in self.ner_annotations:
                ann = self.ner_annotations[tmp]
                spans = ann.get('spans', [])
                
                # Create BIO labels for boundary detection
                if self.use_boundary_labels:
                    bio_labels = self._prepare_bio_labels(data_dict['input_ids'], spans)
                    data_dict['ner_boundary_labels'] = bio_labels
                
                # Create span labels for span classification
                span_labels, candidate_spans = self._prepare_span_labels(
                    data_dict['input_ids'], spans, self.max_span_length
                )
                data_dict['span_labels'] = span_labels
                data_dict['candidate_spans'] = candidate_spans
        
        print(source)
        print(data_dict)
        
        return data_dict


class MultiTaskDataCollator(DataCollatorForSupervisedDataset):
    """
    Data collator for multi-task supervised fine-tuning.
    
    Extends the base data collator to handle auxiliary task targets:
    1. Detection targets - carefully batched to handle variable numbers of objects
    2. Span-based NER targets - spans and boundary labels
    """
    
    def __call__(self, examples):
        """
        Collate examples into batches with proper padding for all tasks.
        
        Adds handling for:
        - Detection targets (boxes, classes)
        - NER targets (boundary labels and span labels)
        """
        # Call parent collator to handle base model inputs
        batch = super().__call__(examples)
        
        # Handle detection targets if present
        has_detection = any('detection_targets' in example for example in examples)
        if has_detection:
            # Collect all detection targets
            all_boxes = []
            all_classes = []
            
            # Find max number of objects across batch
            max_objects = 0
            for example in examples:
                if 'detection_targets' in example:
                    boxes = example['detection_targets']['boxes']
                    max_objects = max(max_objects, boxes.size(0))
            
            # Pad each example to max_objects
            for example in examples:
                if 'detection_targets' in example:
                    boxes = example['detection_targets']['boxes']
                    classes = example['detection_targets']['classes']
                    
                    # Pad or truncate to max_objects
                    num_objects = boxes.size(0)
                    
                    if num_objects < max_objects:
                        # Pad with zeros
                        padded_boxes = torch.zeros((max_objects, 4), dtype=boxes.dtype, device=boxes.device)
                        padded_boxes[:num_objects] = boxes
                        
                        padded_classes = torch.zeros((max_objects,), dtype=classes.dtype, device=classes.device)
                        padded_classes[:num_objects] = classes
                        
                        all_boxes.append(padded_boxes)
                        all_classes.append(padded_classes)
                    else:
                        # Use first max_objects annotations
                        all_boxes.append(boxes[:max_objects])
                        all_classes.append(classes[:max_objects])
                else:
                    # No detection annotations - use zero tensors
                    all_boxes.append(torch.zeros((max_objects, 4), dtype=torch.float32))
                    all_classes.append(torch.zeros((max_objects,), dtype=torch.long))
            
            # Stack into batch tensors
            batch['detection_targets'] = {
                'boxes': torch.stack(all_boxes),
                'classes': torch.stack(all_classes)
            }
        
        # Handle NER boundary labels if present
        has_boundary_labels = any('ner_boundary_labels' in example for example in examples)
        if has_boundary_labels:
            all_boundary_labels = []
            
            for example in examples:
                if 'ner_boundary_labels' in example:
                    all_boundary_labels.append(example['ner_boundary_labels'])
                else:
                    # Use -100 for examples without boundary labels (ignored in loss)
                    all_boundary_labels.append(
                        torch.full_like(example['input_ids'], -100)
                    )
            
            # Pad to same length as input_ids
            batch['ner_boundary_labels'] = self.pad_sequence(
                all_boundary_labels, padding_side='right', padding_value=-100
            )
        
        # Handle span labels if present
        has_span_labels = any('span_labels' in example for example in examples)
        if has_span_labels:
            # For span-based NER, we need to handle span labels and candidate spans
            # This is more complex because number of spans varies per example
            
            max_spans = 0
            for example in examples:
                if 'span_labels' in example:
                    max_spans = max(max_spans, len(example['span_labels']))
            
            # Default span label is -100 (ignored in loss)
            all_span_labels = []
            
            for example in examples:
                if 'span_labels' in example:
                    # Pad to max_spans
                    spans = example['span_labels']
                    padded_spans = torch.full((max_spans,), -100, dtype=spans.dtype, device=spans.device)
                    padded_spans[:len(spans)] = spans
                    all_span_labels.append(padded_spans)
                else:
                    # No spans
                    all_span_labels.append(
                        torch.full((max_spans,), -100, dtype=torch.long)
                    )
            
            batch['span_labels'] = torch.stack(all_span_labels)
            
            # We don't need to batch candidate_spans as they're accessed directly by the model
            # Just store the spans from each example
            batch['candidate_spans'] = [
                example.get('candidate_spans', []) for example in examples
            ]
        
        return batch
    
    def pad_sequence(self, sequences, padding_side='right', padding_value=0):
        """Pad a list of variable length sequences"""
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        
        out_dims = (batch_size, max_len) + trailing_dims
        output = sequences[0].new_full(out_dims, padding_value)
        
        for i, seq in enumerate(sequences):
            length = len(seq)
            if padding_side == 'right':
                output[i, :length] = seq
            else:
                output[i, -length:] = seq
                
        return output


def make_multi_task_data_module(model_id, processor, data_args):
    """
    Create dataset and collator for multi-task fine-tuning.
    
    This factory function prepares:
    1. Multi-task dataset with auxiliary annotations
    2. Collator that handles batching of variable-sized annotations
    """
    dataset = MultiTaskDataset(
        data_path=data_args.data_path, 
        processor=processor, 
        data_args=data_args, 
        model_id=model_id
    )
    
    data_collator = MultiTaskDataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )