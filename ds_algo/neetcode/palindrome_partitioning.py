"""
Palindrome partitioning
https://neetcode.io/problems/palindrome-partitioning

"""

from typing import List

class Solution:
    def get_palindromes(self, s, i, path, res):
        print(s, i, path, res)
        if i >= len(s):
            res.append(path.copy())
        else:
            for j in range(i, len(s)):
                # if palindrome then we have option to take this pattern
                pattern = s[i:j+1]
                if pattern and pattern == pattern[::-1]:
                    path.append(pattern)
                    self.get_palindromes(s, j+1, path, res)
                    path.pop()

    def partition(self, s: str) -> List[List[str]]:
        res = []
        path = []
        self.get_palindromes(s, 0, path, res)
        return res

s = "aab"
sol = Solution()
print(sol.partition(s))