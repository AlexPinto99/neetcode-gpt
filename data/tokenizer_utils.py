from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        t = []
        for num in numbers:
            string = str(num)
            i = 0
            current_tokens = []

            while i < len(string):
                matched = False
                for j in range(len(string), i, -1):
                    substring = string[i:j]

                    if substring in vocab:
                        current_tokens.append(substring)
                        i = j
                        matched = True
                        break
                if not matched:
                    current_tokens.append(string[i])
                    i +=1
            t.append(current_tokens)
        return t


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        string = text
        i = 0
        current_tokens = []

        while i < len(string):
            matched = False
            for j in range(len(string), i, -1):
                substring = string[i:j]

                if substring in vocab:
                    current_tokens.append(substring)
                    i = j
                    matched = True
                    break
            if not matched:
                current_tokens.append(string[i])
                i +=1
        return len(current_tokens)
        

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        token_count = self.count_tokens(text, vocab)
        words = text.split()
        return round(token_count/len(words), 4)
