"""Label encoding utilities"""
import re
from pathlib import Path


class CTCLabelEncoder:
    """Simple CTC label encoder"""

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        max_text_length=25,
    ):
        self.max_text_length = max_text_length
        
        # Load character dict
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        
        # Add blank token for CTC
        dict_character = ["blank"] + dict_character
        
        # Create dict
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def encode(self, text):
        """Encode text to indices"""
        if len(text) == 0:
            return None
        text_list = []
        for char in text:
            if char in self.dict:
                text_list.append(self.dict[char])
        if len(text_list) == 0 or len(text_list) > self.max_text_length:
            return None
        # Pad to max_length
        text_list = text_list + [0] * (self.max_text_length - len(text_list))
        return text_list

