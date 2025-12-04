"""Test validation set and save both correct and incorrect predictions with annotations"""
import json
import string
import subprocess
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from src.data.datamodule import SVTRv2DataModule, collate_fn
from src.models.svtrv2_model import SVTRv2Model
from src.postprocess import build_post_process
from torch.utils.data import DataLoader
from yaml import safe_load


def find_korean_font():
    """Find a Korean font that supports Hangul characters"""
    # Try to find fonts using fc-list (fontconfig)
    font_paths = []
    
    # First, try to use fc-list to find Korean fonts
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    font_path = line.split(":")[0].strip()
                    if font_path and Path(font_path).exists():
                        font_paths.append(font_path)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Add common Korean font paths
    common_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    
    # Add common paths to the list
    for path in common_paths:
        if Path(path).exists() and path not in font_paths:
            font_paths.append(path)
    
    # Try to find fonts in common directories
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "~/.fonts",
        "~/.local/share/fonts",
    ]
    
    for search_dir in search_dirs:
        search_path = Path(search_dir).expanduser()
        if search_path.exists():
            # Look for Korean font files
            for pattern in ["**/*nanum*.ttf", "**/*nanum*.ttc", "**/*noto*cjk*.ttf", "**/*noto*cjk*.ttc"]:
                for font_file in search_path.glob(pattern):
                    if str(font_file) not in font_paths:
                        font_paths.append(str(font_file))
    
    # Test each font to see if it can render Korean characters
    test_text = "가나다라"  # Korean test text
    for font_path in font_paths:
        try:
            test_font = ImageFont.truetype(font_path, 40)
            # Try to get bbox for Korean text
            test_img = Image.new("RGB", (100, 100), "white")
            test_draw = ImageDraw.Draw(test_img)
            bbox = test_draw.textbbox((0, 0), test_text, font=test_font)
            # If bbox is valid, this font likely supports Korean
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                return font_path
        except (OSError, IOError, Exception):
            continue
    
    # If no Korean font found, return None (will use fallback)
    return None


def normalize_text(text, is_filter=False, is_lower=False, ignore_space=True):
    """Normalize text same as RecMetric"""
    if ignore_space:
        text = text.replace(" ", "")
    if is_filter:
        # Keep only alphanumeric characters (same as RecMetric._normalize_text)
        # This removes Korean characters, which matches training metric
        text = "".join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text)
        )
    if is_lower:
        text = text.lower()
    return text


def load_validation_set(config_path):
    """Load validation set from config"""
    with open(config_path, "r") as f:
        config = safe_load(f)
    
    eval_config = config.get("Eval", {})
    dataset_config = eval_config.get("dataset", {})
    
    # Get data directory and jsonl file
    data_dir_list = dataset_config.get("data_dir_list", [])
    if not data_dir_list:
        raise ValueError("No data_dir_list found in Eval config")
    data_dir = Path(data_dir_list[0]).expanduser()
    
    jsonl_file = dataset_config.get("jsonl_file", "val.jsonl")
    jsonl_path = data_dir / jsonl_file
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Validation JSONL file not found: {jsonl_path}")
    
    # Load samples
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            filename = item["filename"]
            text = item["text"]
            
            # Handle relative paths
            if filename.startswith(("imgs/", "train/", "val/")):
                img_path = data_dir / filename
            else:
                img_path = data_dir / "imgs" / filename
                if not img_path.exists():
                    img_path = data_dir / filename
            
            if img_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "label": text,
                    "filename": filename
                })
    
    return samples


# Cache for Korean font path (find once, reuse)
_korean_font_cache = None


def get_korean_font_path():
    """Get Korean font path (cached)"""
    global _korean_font_cache
    if _korean_font_cache is None:
        _korean_font_cache = find_korean_font()
        if _korean_font_cache:
            print(f"Using Korean font: {_korean_font_cache}")
        else:
            print("Warning: No Korean font found, Korean characters may not display correctly")
    return _korean_font_cache


def create_annotated_image(original_img_path, predicted_text, ground_truth_text, output_path):
    """Create image with original + white space + prediction text only"""
    # Load original image
    img = Image.open(original_img_path).convert("RGB")
    orig_width, orig_height = img.size
    
    # Create white background with same width
    text_height = 60  # Height for text area
    white_bg = Image.new("RGB", (orig_width, text_height), color="white")
    
    # Combine images: original on top, white background below
    combined = Image.new("RGB", (orig_width, orig_height + text_height), color="white")
    combined.paste(img, (0, 0))
    combined.paste(white_bg, (0, orig_height))
    
    # Draw text on white background
    draw = ImageDraw.Draw(combined)
    
    # Text to draw (only prediction, no ground truth)
    text_to_draw = predicted_text if predicted_text else "(empty)"
    
    # Get Korean font path (cached)
    available_font_path = get_korean_font_path()
    
    if available_font_path is None:
        # Fallback: try common paths without testing
        fallback_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for font_path in fallback_paths:
            if Path(font_path).exists():
                try:
                    test_font = ImageFont.truetype(font_path, 40)
                    available_font_path = font_path
                    break
                except (OSError, IOError):
                    continue
    
    # Calculate text size and adjust font size to fit within image width
    max_width = orig_width - 20  # Leave 10px margin on each side
    max_height = text_height - 10  # Leave 5px margin on top and bottom
    
    # Find appropriate font size
    font = None
    
    if available_font_path:
        # Try different font sizes to fit the text
        for test_size in range(50, 10, -2):
            try:
                test_font = ImageFont.truetype(available_font_path, test_size)
                text_bbox = draw.textbbox((0, 0), text_to_draw, font=test_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height_actual = text_bbox[3] - text_bbox[1]
                
                if text_width <= max_width and text_height_actual <= max_height:
                    font = test_font
                    break
            except (OSError, IOError):
                continue
    
    # Fallback to default font if no suitable font found
    if font is None:
        font = ImageFont.load_default()
        # For default font, we can't resize, so we'll just use it
        text_bbox = draw.textbbox((0, 0), text_to_draw, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height_actual = text_bbox[3] - text_bbox[1]
        
        # If text is too wide, we might need to truncate (but try to avoid this)
        if text_width > max_width:
            # Try to reduce font size by using a smaller default
            # This is a fallback - ideally we'd have a scalable font
            pass
    
    # Get final text dimensions
    if font != ImageFont.load_default():
        text_bbox = draw.textbbox((0, 0), text_to_draw, font=font)
    else:
        text_bbox = draw.textbbox((0, 0), text_to_draw, font=font)
    
    text_width = text_bbox[2] - text_bbox[0]
    text_height_actual = text_bbox[3] - text_bbox[1]
    
    # Center text horizontally and vertically
    x = max(10, (orig_width - text_width) // 2)  # At least 10px from left edge
    y = orig_height + (text_height - text_height_actual) // 2
    
    # Draw text (always in black for clarity)
    draw.text((x, y), text_to_draw, fill=(0, 0, 0), font=font)
    
    # Save
    combined.save(output_path)
    return combined


def main():
    # Configuration
    config_path = "/home/yjhwang/work/torch-svtrv2/svtrv2/configs/svtrv2_rctc.yml"
    # Try best checkpoint first (lowest val_loss)
    checkpoint_path = "/home/yjhwang/work/torch-svtrv2/svtrv2/output/svtrv2_rctc/1127_v3/checkpoint-epochepoch=77-val_loss=0.442.ckpt"
    base_output_dir = Path("/home/yjhwang/work/torch-svtrv2/svtrv2/output/val_predictions")
    
    # Create separate directories for correct and incorrect predictions
    correct_dir = base_output_dir / "correct"
    incorrect_dir = base_output_dir / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, "r") as f:
        config = safe_load(f)
    
    global_config = config.get("Global", {})
    arch_config = config.get("Architecture", {})
    
    # Resolve character_dict_path before building datamodule
    character_dict_path = global_config.get("character_dict_path")
    if character_dict_path and not Path(character_dict_path).is_absolute():
        config_dir = Path(config_path).parent.parent
        character_dict_path = str(config_dir / character_dict_path.lstrip("./"))
        global_config["character_dict_path"] = character_dict_path
    
    # Build data module (same as training)
    print("Loading validation dataset...")
    datamodule = SVTRv2DataModule.from_config(config)
    datamodule.setup("fit")  # Use "fit" to setup both train and val datasets
    val_dataset = datamodule.val_dataset
    
    if val_dataset is None:
        raise ValueError("Validation dataset not found")
    
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Create DataLoader with same settings as validation (same as training)
    # This ensures images are padded the same way as during training validation
    # Use same batch_size as datamodule (from config)
    eval_config = config.get("Eval", {})
    loader_config = eval_config.get("loader", {})
    val_batch_size = loader_config.get("batch_size_per_card", datamodule.batch_size)
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,  # Use same batch size as validation
        shuffle=False,
        num_workers=datamodule.num_workers,  # Use same num_workers as datamodule
        pin_memory=True,  # Same as datamodule
        drop_last=False,
        collate_fn=collate_fn,  # Use same collate_fn as training
    )
    
    # Build model (same as training)
    print("Initializing model...")
    from src.data.label_encode import CTCLabelEncoder

    # Use already resolved character_dict_path from above
    # It's already an absolute path and updated in global_config
    label_encoder = CTCLabelEncoder(
        character_dict_path=character_dict_path,
        use_space_char=global_config.get("use_space_char", False),
        max_text_length=global_config.get("max_text_length", 25),
    )
    num_classes = len(label_encoder.character)
    
    if "Decoder" in arch_config:
        arch_config["Decoder"]["out_channels"] = num_classes
    else:
        arch_config["Decoder"] = {"out_channels": num_classes}
    
    model = SVTRv2Model(arch_config)
    
    # Load checkpoint (same as infer.py)
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    
    # Lightning checkpoint structure: {'state_dict': {...}, 'epoch': ..., etc.}
    # Or direct state_dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Try loading directly (might be a direct state_dict)
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present (Lightning wraps model in LitModule)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    
    # Load with strict=False to allow partial loading (same as infer.py)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Warning: {len(load_result.missing_keys)} keys were not loaded")
        if len(load_result.missing_keys) > 0:
            print(f"  First 5 missing keys: {list(load_result.missing_keys)[:5]}")
    if load_result.unexpected_keys:
        print(f"Warning: {len(load_result.unexpected_keys)} unexpected keys in checkpoint")
        if len(load_result.unexpected_keys) > 0:
            print(f"  First 5 unexpected keys: {list(load_result.unexpected_keys)[:5]}")
    model = model.cuda()
    model.eval()
    # Ensure all submodules are in eval mode (same as infer.py)
    for module in model.modules():
        module.eval()
    
    # Update global_config with resolved character_dict_path for postprocess (before building)
    # This ensures postprocess uses the correct character dictionary path
    if "character_dict_path" in global_config:
        global_config["character_dict_path"] = character_dict_path
    
    # Build postprocess (same as training and infer.py)
    postprocess_config = config.get("PostProcess", {})
    postprocess = build_post_process(postprocess_config, global_config)
    
    # Process validation set using same pipeline as training
    # Use DataLoader to ensure same preprocessing (padding, etc.) as validation_step
    wrong_count = 0
    correct_count = 0
    exact_match_count = 0  # Count exact matches (without normalization)
    saved_correct_count = 0  # Count of saved correct predictions (max 5)
    saved_incorrect_count = 0  # Count of saved incorrect predictions
    max_correct_save_count = 5  # Maximum number of correct predictions to save
    
    print("Processing validation set...")
    print(f"Will save up to {max_correct_save_count} correct predictions and all incorrect predictions...")
    
    # Debug: Check first few predictions
    debug_count = 0
    
    # Process using DataLoader (same as validation_step)
    for batch_idx, batch in enumerate(val_dataloader):
        # Get batch data (same format as validation_step)
        images = batch["image"].cuda()  # Already batched and padded by collate_fn
        labels = batch["label"]  # List of label strings
        batch_size = len(labels)
        
        # Forward pass (same as validation_step)
        # validation_step does train() -> eval() for loss calculation,
        # but post-processing uses eval() mode output
        # We need to match this behavior exactly
        with torch.no_grad():
            # First forward pass in train mode (like validation_step for loss)
            # This may affect internal state (e.g., BatchNorm running stats)
            model.train()
            _ = model(images)
            # Then switch to eval mode for post-processing (same as validation_step)
            model.eval()
            preds = model(images)
        
        # Check for invalid model output (same as infer.py)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print(f"Warning: NaN or Inf detected in model output for batch {batch_idx+1}")
            pred_texts = [("", 0.0)] * batch_size
        else:
            # Post-process (same as validation_step)
            pred_texts = postprocess(preds, batch=None, torch_tensor=True)
        
        # Process each sample in batch
        for batch_item_idx in range(batch_size):
            # Get original image path for saving
            sample_idx = batch_idx * val_batch_size + batch_item_idx
            if sample_idx >= len(val_dataset):
                break
            original_img_path = val_dataset.samples[sample_idx]["image_path"]
            filename = Path(original_img_path).name
            ground_truth = labels[batch_item_idx]
            
            # Get prediction for this item in batch
            if batch_item_idx < len(pred_texts):
                pred_text, score = pred_texts[batch_item_idx]
            else:
                pred_text = ""
                score = 0.0
            
            # Debug: Print first 10 predictions with more details
            if debug_count < 10:
                pred_normalized = normalize_text(pred_text, is_filter=True, is_lower=True, ignore_space=True)
                gt_normalized = normalize_text(ground_truth, is_filter=True, is_lower=True, ignore_space=True)
                match = "✓" if pred_normalized == gt_normalized else "✗"
                print(f"  Sample {sample_idx+1}: {match} GT='{ground_truth}' -> Pred='{pred_text}' (norm: '{gt_normalized}' == '{pred_normalized}') Score={score:.4f}")
                debug_count += 1
            
            # Check both exact match and normalized match
            exact_match = pred_text == ground_truth
            if exact_match:
                exact_match_count += 1
            
            # Normalize texts same as validation metric (is_filter=True, is_lower=True, ignore_space=True)
            # This matches the metric used during training
            pred_normalized = normalize_text(pred_text, is_filter=True, is_lower=True, ignore_space=True)
            gt_normalized = normalize_text(ground_truth, is_filter=True, is_lower=True, ignore_space=True)
            normalized_match = pred_normalized == gt_normalized
            
            # Create output filename (sanitize for filesystem)
            safe_filename = Path(filename).stem
            # Replace problematic characters in text
            safe_pred = pred_text.replace("/", "_").replace("\\", "_")[:20]
            safe_gt = ground_truth.replace("/", "_").replace("\\", "_")[:20]
            output_filename = f"{safe_filename}_pred_{safe_pred}_gt_{safe_gt}.jpg"
            
            # Use normalized comparison (same as training validation metric)
            if normalized_match:
                correct_count += 1
                
                # Save only if we haven't saved enough correct predictions yet
                if saved_correct_count < max_correct_save_count:
                    saved_correct_count += 1
                    output_path = correct_dir / output_filename
                    
                    # Create annotated image (show only prediction text)
                    create_annotated_image(
                        original_img_path,
                        pred_text,
                        ground_truth,  # Passed but not displayed
                        output_path
                    )
                    
                    print(f"  Saved correct prediction {saved_correct_count}/{max_correct_save_count}: {output_filename}")
            else:
                wrong_count += 1
                saved_incorrect_count += 1
                
                output_path = incorrect_dir / output_filename
                
                # Create annotated image (show only prediction text)
                create_annotated_image(
                    original_img_path,
                    pred_text,
                    ground_truth,  # Passed but not displayed
                    output_path
                )
                
                if saved_incorrect_count % 10 == 0:
                    print(f"  Saved {saved_incorrect_count} incorrect predictions...")
        
        total_processed = min((batch_idx + 1) * val_batch_size, len(val_dataset))
        if total_processed % 100 == 0 or (batch_idx + 1) == len(val_dataloader):
            print(f"  Processed {total_processed}/{len(val_dataset)} samples...")
    
    print("\nResults:")
    print(f"  Total samples: {len(val_dataset)}")
    print(f"  Correct (normalized): {correct_count}")
    print(f"  Wrong (normalized): {wrong_count}")
    print(f"  Exact matches: {exact_match_count}")
    print(f"  Saved correct predictions: {saved_correct_count}/{max_correct_save_count}")
    print(f"  Saved incorrect predictions: {saved_incorrect_count}")
    if len(val_dataset) > 0:
        print(f"  Accuracy (normalized, same as training): {correct_count / len(val_dataset) * 100:.2f}%")
        print(f"  Accuracy (exact match): {exact_match_count / len(val_dataset) * 100:.2f}%")
    print(f"\n  Correct predictions saved to: {correct_dir}")
    print(f"  Incorrect predictions saved to: {incorrect_dir}")
    
    # Print sample of saved predictions
    if saved_correct_count > 0:
        print("\nSample correct predictions (first 5):")
        correct_files = sorted(correct_dir.glob("*.jpg"))[:saved_correct_count]
        for f in correct_files:
            print(f"  {f.name}")
    
    if saved_incorrect_count > 0:
        print("\nSample incorrect predictions (first 5):")
        incorrect_files = sorted(incorrect_dir.glob("*.jpg"))[:5]
        for f in incorrect_files:
            print(f"  {f.name}")
    
    # Print some statistics
    print("\nNote: Comparison uses normalized text (is_filter=True, is_lower=True)")
    print("This matches the validation metric used during training.")


if __name__ == "__main__":
    main()

