from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = [char for char in corpus]
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            pairs ={}
            for i in range(len(tokens)-1):


                pair = (tokens[i], tokens[i+1])

                if pair in pairs:
                    pairs[pair] +=1
                else:
                    pairs[pair] = 1

            highest = max(sorted(pairs.keys()), key=pairs.get)

            new_tokens = []
            merges.append(list(highest))

            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == highest:
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        
        return merges
















