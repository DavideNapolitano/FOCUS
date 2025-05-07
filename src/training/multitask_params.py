# src/training/params.py
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768,  # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    
    # Multi-task learning parameters
    
    # Object detection parameters
    include_detection: bool = field(
        default=False,
        metadata={"help": "Whether to include object detection as an auxiliary task"}
    )
    detection_annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to COCO-format detection annotations"}
    )
    num_object_classes: int = field(
        default=80,  # COCO has 80 classes
        metadata={"help": "Number of object classes to detect"}
    )
    
    # Layer selection parameters (percentage-based approach)
    vision_layer_percentage: Optional[float] = field(
        default=0.5,  # 50% by default
        metadata={"help": "Percentage of vision layers to use for object detection (0.0-1.0)"}
    )
    language_layer_percentage: Optional[float] = field(
        default=0.5,  # 50% by default
        metadata={"help": "Percentage of language model layers to use for NER (0.0-1.0)"}
    )
    
    # Legacy absolute layer indices (still supported but percentage-based is preferred)
    object_detection_layer: int = field(
        default=11,
        metadata={"help": "Specific vision layer to use for object detection (ignored if vision_layer_percentage is set)"}
    )
    ner_layer: int = field(
        default=11,
        metadata={"help": "Specific language model layer to use for NER (ignored if language_layer_percentage is set)"}
    )
    
    detection_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for the object detection loss component"}
    )
    
    # NER parameters
    include_ner: bool = field(
        default=False,
        metadata={"help": "Whether to include named entity recognition as an auxiliary task"}
    )
    ner_annotation_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to NER annotations"}
    )
    num_entity_types: int = field(
        default=10,
        metadata={"help": "Number of entity types to recognize"}
    )
    ner_loss_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for the NER loss component"}
    )
    
    # Span-based NER parameters
    max_span_length: int = field(
        default=8,
        metadata={"help": "Maximum token length for entity spans"}
    )
    use_boundary_labels: bool = field(
        default=True,
        metadata={"help": "Whether to use BIO tagging scheme for boundary detection"}
    )
    
    # Cross-modal alignment
    contrastive_loss_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the contrastive loss between visual and textual features"}
    )