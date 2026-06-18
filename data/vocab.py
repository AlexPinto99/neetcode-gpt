from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        unique_chars = [char for char in set(text)]
        unique_chars=sorted(unique_chars)
        stoi={}
        itos={}
        for index, char in enumerate(unique_chars):
            stoi[char] = index
            itos[index] = char
        return (stoi, itos)



    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded = [stoi[char] for char in text ]
        return encoded

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded = [itos[num] for num in ids]
        unified_string=""
        unified_string = unified_string.join(decoded)
        return unified_string 
