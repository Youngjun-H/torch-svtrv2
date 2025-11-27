"""CTC Post-processing"""
import re

import numpy as np


class BaseRecLabelDecode:
    """Base class for label decoding"""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        """Add special characters"""
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """Convert text-index into text-label"""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        num_chars = len(self.character)
        
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            
            # Filter out invalid indices (out of range)
            valid_selection = selection.copy()
            text_indices = text_index[batch_idx]
            valid_selection &= (text_indices >= 0) & (text_indices < num_chars)
            
            # Build character list with valid indices only
            char_list = []
            valid_indices = text_indices[valid_selection]
            if len(valid_indices) == 0:
                # Debug: Check why no valid indices
                if batch_idx == 0:  # Only print for first batch to avoid spam
                    print(f"Debug - No valid indices after filtering")
                    print(f"Debug - Original text_index length: {len(text_index[batch_idx])}")
                    print(f"Debug - Selection after duplicate removal: {selection.sum()}")
                    print(f"Debug - Selection after ignored tokens: {(text_indices != ignored_tokens[0]).sum()}")
                    print(f"Debug - Valid selection: {valid_selection.sum()}")
                    unique_indices = list(set(text_indices.tolist()))[:10]
                    print(f"Debug - Unique indices in original: {unique_indices}")
                    print(f"Debug - All indices are blank (0)? {all(idx == 0 for idx in text_indices)}")
                # If no valid indices, return empty string
                result_list.append(("", 0.0))
                continue
            for text_id in valid_indices:
                if 0 <= text_id < num_chars:
                    char_list.append(self.character[text_id])
                else:
                    # Skip invalid indices
                    continue
            
            if text_prob is not None:
                conf_list = text_prob[batch_idx][valid_selection]
            else:
                conf_list = [1] * len(char_list)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        
        return result_list

    def get_ignored_tokens(self):
        """Get ignored tokens"""
        return [0]  # for ctc blank

    def get_character_num(self):
        """Get number of characters"""
        return len(self.character)


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC Label Decode"""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, batch=None, **kwargs):
        """Decode predictions"""
        if kwargs.get("torch_tensor", True):
            preds = preds.detach().cpu().numpy()
        
        # Ensure preds is in probability space (not log space)
        # RCTCDecoder applies softmax in eval mode, but check if needed
        if preds.ndim == 3:
            # Check if values sum to ~1 (already probabilities) or are logits
            import numpy as np
            sample_sum = np.sum(preds[0, 0, :])
            if sample_sum < 0.9 or sample_sum > 1.1:
                # Likely logits, apply softmax
                if preds.min() < 0:
                    preds = np.exp(preds - np.max(preds, axis=2, keepdims=True))
                    preds = preds / (np.sum(preds, axis=2, keepdims=True) + 1e-8)
        
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        
        if batch is None:
            return text
        
        # Decode labels if batch is provided
        if isinstance(batch, dict):
            # Handle dict format - labels are strings, need to convert to indices
            labels = batch.get("label", [])
            label_list = []
            for label_str in labels:
                label_indices = [self.dict.get(c, 0) for c in label_str]
                label_list.append((label_str, 1.0))
            return text, label_list
        else:
            label = self.decode(batch[1])
            return text, label

    def add_special_char(self, dict_character):
        """Add blank token for CTC"""
        dict_character = ["blank"] + dict_character
        return dict_character

