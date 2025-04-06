"""
https://neetcode.io/problems/jump-game
"""

from typing import List


# using dp solution is O(n2)

class Solution:
    def canjumpdp(self, nums, i, mapping):
        if i == len(nums) - 1:
            mapping[i] = True
            return mapping
        jump_len = nums[i]
        if jump_len == 0:
            mapping[i] = False
            print(i, mapping)
            return mapping
        jump_span = i+1, min(len(nums), i + jump_len + 1)
        print(f'{i}, {jump_span=}')
        for j in range(jump_span[0], jump_span[1]):
            if j == len(nums) - 1:
                print(i, mapping)
                mapping[i] = True
            elif i != j and j not in mapping: # since there is no point in not jumping
                mapping = self.canjumpdp(nums, j, mapping)
                mapping[i] = mapping.get(i, False) or mapping[j]
            else:
                pass
        return mapping

    def canJump(self, nums: List[int]) -> bool:
        mapping = {}
        mapping = self.canjumpdp(nums, 0, mapping)
        return mapping[0]


# greedy solution

class Solution:
    def can_reach(self, nums, i):
        for j in range(i-1, -1, -1):
            jump_range = nums[j]
            if j + jump_range >= i:
                return j

    def canJump(self, nums: List[int]) -> bool:
        # greedy approach, 2 pointer
        goal = len(nums) - 1
        while goal > 0:
            goal = self.can_reach(nums, goal)
            if goal is None:
                return False
        return True



nums = [1,2,0,1,0]
print(f'{nums=}')
s = Solution()
out = s.canJump(nums)
print(out)