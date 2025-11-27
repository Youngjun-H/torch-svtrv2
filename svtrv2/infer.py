"""Inference script for SVTRv2 text recognition"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from src.data.label_encode import CTCLabelEncoder
from src.models.svtrv2_model import SVTRv2Model
from src.postprocess import build_post_process
from torchvision import transforms as T
from torchvision.transforms import functional as F
from yaml import safe_load


class SVTRv2Inference:
    """SVTRv2 Inference class"""

    def __init__(self, config_path, checkpoint_path, device="cuda"):
        """
        Args:
            config_path: Path to config YAML file
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = safe_load(f)
        
        self.global_config = self.config.get("Global", {})
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load character dictionary
        character_dict_path = self.global_config.get("character_dict_path")
        use_space_char = self.global_config.get("use_space_char", False)
        max_text_length = self.global_config.get("max_text_length", 25)
        
        self.label_encoder = CTCLabelEncoder(
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            max_text_length=max_text_length,
        )
        num_classes = len(self.label_encoder.character)
        
        # Build model
        arch_config = self.config.get("Architecture", {})
        if "Decoder" in arch_config:
            arch_config["Decoder"]["out_channels"] = num_classes
        else:
            arch_config["Decoder"] = {"out_channels": num_classes}
        
        self.model = SVTRv2Model(arch_config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Remove 'model.' prefix if present (Lightning checkpoint)
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            # Try loading directly
            state_dict = checkpoint
            # Remove 'model.' prefix if present
            if any(k.startswith("model.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        # Check which keys match
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        print(f"Debug - Loading checkpoint from: {checkpoint_path}")
        print(f"Debug - Model has {len(model_keys)} parameters")
        print(f"Debug - Checkpoint has {len(checkpoint_keys)} parameters")
        print(f"Debug - Missing keys: {len(missing_keys)}")
        print(f"Debug - Unexpected keys: {len(unexpected_keys)}")
        if missing_keys:
            print(f"Debug - First 10 missing keys: {list(missing_keys)[:10]}")
        if unexpected_keys:
            print(f"Debug - First 10 unexpected keys: {list(unexpected_keys)[:10]}")
        
        # Load with strict=False to allow partial loading
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            print(f"Warning: {len(load_result.missing_keys)} keys were not loaded")
        if load_result.unexpected_keys:
            print(f"Warning: {len(load_result.unexpected_keys)} unexpected keys in checkpoint")
        
        self.model.to(self.device)
        self.model.eval()
        # Ensure all submodules are in eval mode
        for module in self.model.modules():
            module.eval()
        
        # Debug flag for first inference
        self._first_inference = True
        
        # Build post-process
        postprocess_config = self.config.get("PostProcess", {})
        self.postprocess = build_post_process(postprocess_config, self.global_config)
        
        # Image preprocessing
        # Try to get from Eval config, fallback to Train config, then defaults
        eval_config = self.config.get("Eval", {})
        train_config = self.config.get("Train", {})
        self.base_h = (
            eval_config.get("dataset", {}).get("base_h") or
            train_config.get("dataset", {}).get("base_h") or
            32
        )
        self.base_shape = (
            eval_config.get("dataset", {}).get("base_shape") or
            train_config.get("dataset", {}).get("base_shape") or
            [[64, 32], [96, 32], [128, 32], [160, 32]]
        )
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.interpolation = T.InterpolationMode.BICUBIC

    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            # Assume numpy array
            image = Image.fromarray(image).convert("RGB")
        
        w, h = image.size
        
        # Resize to target height while maintaining aspect ratio
        target_h = self.base_h
        target_w = int(w * target_h / h)
        
        # Resize
        image = F.resize(image, (target_h, target_w), interpolation=self.interpolation)
        
        # Limit max width
        max_w = max([s[0] for s in self.base_shape]) if self.base_shape else None
        if max_w and target_w > max_w:
            image = F.crop(image, 0, 0, target_h, max_w)
            target_w = max_w
        
        # Normalize
        image = self.normalize(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image

    def predict(self, image, batch_size=1):
        """
        Predict text from image
        
        Args:
            image: PIL Image, numpy array, or image path
            batch_size: Batch size (for multiple images)
            
        Returns:
            List of (text, score) tuples
        """
        # Handle single image or list of images
        if isinstance(image, (str, Path)) or not isinstance(image, list):
            images = [image]
        else:
            images = image
        
        results = []
        
        for img in images:
            # Preprocess
            img_tensor = self.preprocess_image(img)
            img_tensor = img_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                preds = self.model(img_tensor)
            
            # Debug: Check model output (only for first image to avoid spam)
            if len(results) == 0:
                print(f"Debug - Model output shape: {preds.shape}")
                print(f"Debug - Model output min/max: {preds.min().item():.4f} / {preds.max().item():.4f}")
                print(f"Debug - Model output mean: {preds.mean().item():.4f}")
                
                # Debug: Check if model is in eval mode
                print(f"Debug - Model training mode: {self.model.training}")
                if hasattr(self.model, 'decoder'):
                    print(f"Debug - Decoder training mode: {self.model.decoder.training}")
                
                # Debug: Check argmax predictions
                preds_np = preds.detach().cpu().numpy()
                preds_idx_debug = preds_np.argmax(axis=2)
                unique_indices = sorted(set(preds_idx_debug.flatten()))
                print(f"Debug - Unique predicted indices: {unique_indices[:10]}")  # Show first 10
                print(f"Debug - Most common index: {max(set(preds_idx_debug.flatten()), key=list(preds_idx_debug.flatten()).count)}")
                
                # Debug: Check probability distribution
                print(f"Debug - Probability sum (should be ~1.0): {preds_np[0, 0, :].sum():.4f}")
                print(f"Debug - Max probability at each position: {preds_np[0, :, :].max(axis=1)[:5]}")  # First 5 positions
            
            # Debug: Check model output
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                if isinstance(img, (str, Path)):
                    img_name = str(img)
                else:
                    img_name = "image"
                print(f"Warning: NaN or Inf detected in model output for {img_name}")
                results.append(("", 0.0))
                continue
            
            # Post-process
            pred_texts = self.postprocess(preds, batch=None, torch_tensor=True)
            
            results.extend(pred_texts)
        
        return results

    def predict_batch(self, image_list, batch_size=8):
        """
        Predict text from a batch of images
        
        Args:
            image_list: List of images (PIL Image, numpy array, or paths)
            batch_size: Batch size
            
        Returns:
            List of (text, score) tuples
        """
        # Preprocess all images
        batch_images = []
        max_width = 0
        max_height = 0
        
        for img in image_list:
            img_tensor = self.preprocess_image(img)
            _, _, h, w = img_tensor.shape
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            batch_images.append(img_tensor.squeeze(0))
        
        # Pad to same size
        padded_batch = []
        for img in batch_images:
            _, h, w = img.shape
            pad_w = max_width - w
            pad_h = max_height - h
            if pad_w > 0 or pad_h > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            padded_batch.append(img)
        
        # Stack
        batch_tensor = torch.stack(padded_batch, dim=0).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds = self.model(batch_tensor)
        
        # Post-process
        pred_texts = self.postprocess(preds, batch=None, torch_tensor=True)
        
        return pred_texts


def main():
    parser = argparse.ArgumentParser(description="SVTRv2 Text Recognition Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file or directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./inference_results.txt",
        help="Path to output file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = SVTRv2Inference(args.config, args.checkpoint, args.device)
    
    # Get image files
    image_path = Path(args.image)
    if image_path.is_file():
        image_files = [image_path]
    elif image_path.is_dir():
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")) + list(image_path.glob("*.jpeg"))
    else:
        raise ValueError(f"Invalid image path: {args.image}")
    
    # Run inference
    results = []
    for img_file in image_files:
        pred_texts = inferencer.predict(str(img_file), batch_size=args.batch_size)
        for pred_text, score in pred_texts:
            results.append({
                "file": str(img_file),
                "text": pred_text,
                "score": score,
            })
            print(f"{img_file}: {pred_text} (score: {score:.4f})")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result['file']}\t{result['text']}\t{result['score']:.4f}\n")
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

