"""
https://neetcode.io/problems/longest-substring-without-duplicates
"""


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        using two pointer we can create the solution
        """
        if not s:
            return 0
        uniq_chrs = {s[0]: True}
        longest = 1
        i = 0
        for j in range(1, len(s)):
            incoming_ch  = s[j]
            if incoming_ch in uniq_chrs:
                while i < j and incoming_ch in uniq_chrs:
                    outgoing_ch = s[i]
                    del uniq_chrs[outgoing_ch]
                    i += 1
            uniq_chrs[incoming_ch] = True
            longest = max(longest, len(uniq_chrs))
        return max(longest, len(uniq_chrs))
    

s="abcabcbb"
sol = Solution()
print(sol.lengthOfLongestSubstring(s))