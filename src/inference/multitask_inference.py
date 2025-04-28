# src/inference/multitask_inference.py
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
from peft import PeftModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Union, Tuple, Optional

from src.utils import load_pretrained_model, get_model_name_from_path
from src.training.multi_task_model import MultiTaskQwen2VL, PercentageBasedDetectionHead, NERHead


def load_multitask_model(
    model_path: str,
    model_base: Optional[str] = None,
    device: str = "cuda",
    load_8bit: bool = False,
    load_4bit: bool = False,
):
    """
    Load a multi-task model for inference.
    
    Args:
        model_path: Path to the saved model
        model_base: Base model name if using LoRA
        device: Device to load the model on
        load_8bit: Whether to load in 8-bit precision
        load_4bit: Whether to load in 4-bit precision
        
    Returns:
        processor: The model processor
        model: The loaded multi-task model
    """
    # First load the base VLM
    processor, base_model = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        device_map=device,
        load_8bit=load_8bit,
        load_4bit=load_4bit
    )
    
    # Try to load auxiliary task configuration
    auxiliary_config_path = os.path.join(model_path, "auxiliary_config.json")
    auxiliary_heads_path = os.path.join(model_path, "auxiliary_heads.bin")
    
    # Default configuration
    config = {
        "object_detection_layer": 11,
        "ner_layer": 11,
        "num_object_classes": 80,
        "num_entity_types": 10,
        "det_loss_weight": 0.5,
        "ner_loss_weight": 0.5,
        "contrastive_loss_weight": 0.2
    }
    
    # Load configuration if available
    if os.path.exists(auxiliary_config_path):
        import json
        with open(auxiliary_config_path, 'r') as f:
            config.update(json.load(f))
    
    # Create multi-task model
    model = MultiTaskQwen2VL(
        base_model=base_model,
        object_detection_layer=config["object_detection_layer"],
        ner_layer=config["ner_layer"],
        num_object_classes=config["num_object_classes"],
        num_entity_types=config["num_entity_types"],
        det_loss_weight=config["det_loss_weight"],
        ner_loss_weight=config["ner_loss_weight"],
        contrastive_loss_weight=config["contrastive_loss_weight"]
    )
    
    # Load auxiliary heads if available
    if os.path.exists(auxiliary_heads_path):
        auxiliary_dict = torch.load(auxiliary_heads_path, map_location="cpu")
        
        if "detection_head" in auxiliary_dict:
            model.detection_head.load_state_dict(auxiliary_dict["detection_head"])
            
        if "ner_head" in auxiliary_dict:
            model.ner_head.load_state_dict(auxiliary_dict["ner_head"])
            
        # Update configuration if available
        if "config" in auxiliary_dict:
            config.update(auxiliary_dict["config"])
    
    # Set model to evaluation mode
    model.eval()
    
    return processor, model


def visualize_detections(
    image: Union[Image.Image, str],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    score_threshold: float = 0.5,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Visualize object detections on an image.
    
    Args:
        image: PIL Image or path to image
        boxes: Tensor of box coordinates in [0, 1] range
        scores: Tensor of confidence scores
        labels: Tensor of class indices
        class_names: List of class names
        score_threshold: Minimum score to display a detection
        output_path: Path to save the visualization
        show: Whether to display the visualization
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    # Convert to numpy
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Prepare figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(image))
    
    # Get image dimensions
    width, height = image.size
    
    # Define colors for different classes (using HSV for better distribution)
    num_classes = len(class_names)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes))
    
    # Draw each detection
    for box, score, label in zip(boxes_np, scores_np, labels_np):
        # Skip low-confidence detections
        if score < score_threshold:
            continue
        
        # Convert normalized coordinates to pixel values
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=colors[label],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Create label text
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Draw label background
        text_box = ax.text(
            x1, y1-5, label_text,
            bbox=dict(facecolor=colors[label], alpha=0.5),
            fontsize=10, color='white'
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def highlight_entities(
    text: str,
    ner_predictions: torch.Tensor,
    entity_types: List[str],
    tokenizer,
    confidence_threshold: float = 0.5,
):
    """
    Format text with highlighted entities based on NER predictions.
    
    Args:
        text: The input text
        ner_predictions: Tensor of NER logits [seq_len, num_entity_types]
        entity_types: List of entity type names
        tokenizer: The tokenizer used to process the text
        confidence_threshold: Minimum confidence to highlight an entity
        
    Returns:
        html_text: Text with HTML highlighting for entities
    """
    # Convert logits to probabilities
    probs = torch.softmax(ner_predictions, dim=-1)
    
    # Get the highest probability entity for each token
    best_entity_scores, best_entity_ids = torch.max(probs, dim=-1)
    
    # Convert tensors to numpy
    best_entity_scores = best_entity_scores.cpu().numpy()
    best_entity_ids = best_entity_ids.cpu().numpy()
    
    # Define colors for different entity types (using HSV for better distribution)
    num_entities = len(entity_types)
    colors = plt.cm.hsv(np.linspace(0, 1, num_entities))
    colors_hex = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in colors]
    
    # Tokenize text to align predictions with tokens
    tokens = tokenizer.tokenize(text)
    
    # Create HTML with highlighted entities
    html_parts = []
    current_entity = None
    current_score = 0
    
    for i, (token, entity_id, score) in enumerate(zip(tokens, best_entity_ids, best_entity_scores)):
        # Skip non-entity tokens (usually 0 is "O" or outside any entity)
        if entity_id == 0 or score < confidence_threshold:
            if current_entity is not None:
                # Close previous entity span
                html_parts.append(f"</span>")
                current_entity = None
            
            # Add regular token
            token_text = tokenizer.convert_tokens_to_string([token])
            html_parts.append(token_text)
        else:
            entity_type = entity_types[entity_id] if entity_id < len(entity_types) else f"Entity{entity_id}"
            
            if current_entity != entity_type:
                # Close previous entity if exists
                if current_entity is not None:
                    html_parts.append(f"</span>")
                
                # Start new entity span
                color = colors_hex[entity_id % len(colors_hex)]
                html_parts.append(f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{entity_type} ({score:.2f})">')
                current_entity = entity_type
                current_score = score
            
            # Add token within entity span
            token_text = tokenizer.convert_tokens_to_string([token])
            html_parts.append(token_text)
    
    # Close any open span
    if current_entity is not None:
        html_parts.append("</span>")
    
    return "".join(html_parts)


def run_multitask_inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    class_names: List[str],
    entity_types: List[str],
    detection_score_threshold: float = 0.5,
    ner_confidence_threshold: float = 0.5,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """
    Run inference with a multi-task model, performing:
    1. Regular VLM image understanding (main task)
    2. Object detection with visualizations (auxiliary task)
    3. Named entity recognition in the output text (auxiliary task)
    
    Args:
        model: The multi-task model
        processor: The model processor
        image_path: Path to the input image
        prompt: User prompt about the image
        class_names: List of object class names
        entity_types: List of entity type names
        detection_score_threshold: Threshold for detection confidence
        ner_confidence_threshold: Threshold for NER confidence
        temperature: Sampling temperature
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        dict: Dictionary containing the VLM response, detection results, and NER results
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    # Process input
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.base_model.device)
    
    # Prepare image for model
    image_inputs, _ = process_vision_info(messages)
    if image_inputs:
        inputs_dict = processor(
            images=image_inputs,
            return_tensors="pt"
        ).to(model.base_model.device)
        pixel_values = inputs_dict.get("pixel_values")
        image_grid_thw = inputs_dict.get("image_grid_thw")
    
    # Generate response
    with torch.no_grad():
        # Forward pass through model
        outputs = model(
            input_ids=inputs,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
            output_hidden_states=True
        )
        
        # Extract detections from intermediate features
        detection_classes, detection_boxes, attention_weights = model.detection_head(model.vision_features)
        
        # Get detection confidence scores
        detection_scores = torch.softmax(detection_classes, dim=-1)
        # Get the highest non-background class (assume background is class 0)
        best_scores, best_classes = detection_scores[:, :, 1:].max(dim=-1)
        # Adjust class indices (add 1 since we excluded background)
        best_classes = best_classes + 1
        
        # Get response generation
        generated_ids = model.base_model.generate(
            input_ids=inputs,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
        )
        
        # Decode generated text
        generated_text = processor.tokenizer.decode(
            generated_ids[0, inputs.shape[1]:], skip_special_tokens=True
        )
        
        # Tokenize the generated text for NER
        generated_inputs = processor.tokenizer(generated_text, return_tensors="pt").to(model.base_model.device)
        
        # Get text features for NER by running another forward pass on just the generated text
        text_outputs = model.base_model(
            input_ids=generated_inputs.input_ids,
            attention_mask=generated_inputs.attention_mask,
            output_hidden_states=True
        )
        
        # Extract features from the specified layer
        text_layer_outputs = text_outputs.hidden_states[model.ner_layer]
        
        # Run NER head on text features
        ner_logits = model.ner_head(text_layer_outputs)
    
    # Visualize detections
    detection_output_path = os.path.splitext(image_path)[0] + "_detections.jpg"
    visualize_detections(
        image=image,
        boxes=detection_boxes[0],  # First batch item
        scores=best_scores[0],     # First batch item
        labels=best_classes[0],    # First batch item
        class_names=class_names,
        score_threshold=detection_score_threshold,
        output_path=detection_output_path,
        show=False
    )
    
    # Highlight entities in generated text
    highlighted_text = highlight_entities(
        text=generated_text,
        ner_predictions=ner_logits[0],  # First batch item
        entity_types=entity_types,
        tokenizer=processor.tokenizer,
        confidence_threshold=ner_confidence_threshold
    )
    
    # Return all results
    return {
        "prompt": prompt,
        "response": generated_text,
        "highlighted_response": highlighted_text,
        "detection_visualization_path": detection_output_path,
        "detections": {
            "boxes": detection_boxes[0].cpu(),
            "scores": best_scores[0].cpu(),
            "classes": best_classes[0].cpu()
        },
        "ner_logits": ner_logits[0].cpu()
    }


# Example usage
if __name__ == "__main__":
    # Example class names (COCO classes)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Example entity types
    ENTITY_TYPES = [
        "O",           # Outside any entity
        "PERSON",      # People, including fictional
        "LOCATION",    # Physical locations
        "ORGANIZATION",# Companies, agencies, institutions
        "DATE",        # Absolute or relative dates
        "TIME",        # Times smaller than a day
        "MONEY",       # Monetary values
        "PERCENT",     # Percentage
        "PRODUCT",     # Products and objects
        "EVENT"        # Named events
    ]
    
    # Load model
    processor, model = load_multitask_model(
        model_path="/path/to/your/multitask_model",
        device="cuda"
    )
    
    # Run inference
    results = run_multitask_inference(
        model=model,
        processor=processor,
        image_path="/path/to/your/image.jpg",
        prompt="Describe what you see in this image, including any people and objects.",
        class_names=COCO_CLASSES,
        entity_types=ENTITY_TYPES
    )
    
    # Print results
    print(f"Prompt: {results['prompt']}")
    print(f"Response: {results['response']}")
    print(f"Detection visualization saved to: {results['detection_visualization_path']}")
    print(f"HTML response with highlighted entities is available in results['highlighted_response']")