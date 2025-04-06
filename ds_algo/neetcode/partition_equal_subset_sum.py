"""
https://neetcode.io/problems/partition-equal-subset-sum

this is similar to find subarray with target

"""

from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # total sum must be an even number. odd numbers then we cannot divide.
        # sorting and then using two pointers.
        # that way this can be done in O(n) time
        # this works because the numbers are positive
        # the two pointer solution is not correct
        nums.sort()
        print(nums)
        i = 0
        j = 1
        total = sum(nums)
        if total % 2 != 0:
            return False
        total = total / 2
        curr = nums[i]
        while i < j and j < len(nums):
            print(curr, i, j, total)
            if curr == total:
                return True
            elif curr < total:
                while j < len(nums) and curr < total:
                    jval = nums[j]
                    curr += jval
                    j += 1
                    if curr == total:
                        return True
            else:
                while i < j and curr > total:
                    ival = nums[i]
                    curr -= ival
                    i += 1
                    if curr == total:
                        return True
        return False



class Solution:
    calls = 1
    def can_partition_rec(self, nums, i, target, dp):
        self.calls += 1
        # base cases
        if target < 0 or i < 0:
            return False
        if target == 0:
            return True
        if i == 0 and nums[i] == target:
            return True
        if (i, target) in dp:
            return dp[(i, target)]
        out = self.can_partition_rec(nums, i-1, target, dp)
        dp[(i, target)] = out
        if out is True:
            return True
        if nums[i] <= target:
            out = self.can_partition_rec(nums, i-1, target-nums[i], dp)
        dp[(i, target)] = out
        return out

    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        total = total / 2
        dp = {}
        return self.can_partition_rec(nums, len(nums)-1, total, dp)

import numpy as np

# good explanation of tabulation https://www.youtube.com/watch?v=fWX9xDmIzRI&t=1364s

class Solution:
    # this is the tabulation dp solution
    calls = 1
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        total = total // 2
        dp = [[False for _ in range(total+1)] for _ in range(len(nums))]
        # base cases
        for i in range(len(nums)):
            dp[i][0] = True
        dp[0][nums[0]] = True
        dp = np.array(dp)
        print('startin dp')
        print(dp)
        for i in range(len(nums)):
            for t in range(1, total+1):
                dp[i][t] = (
                    dp[i-1][t-nums[i]] # if the value is considere
                    or dp[i-1][t] # if the value is not considered to calculate the target
                )
        dp = np.array(dp)
        print('final dp')
        print(dp)
        return dp[len(nums)-1][total]


# with space optimisation for tabulation
class Solution:
    # this is the tabulation dp solution
    calls = 1
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        total = total // 2
        # one for previous and one for current
        dp = [[False for _ in range(total+1)] for _ in range(2)]
        # base cases
        dp[0][0] = True
        dp[1][0] = True
        dp[0][nums[0]] = True
        dp = np.array(dp)
        print('startin dp')
        print(dp)
        curr = 1
        prev = 0
        for i in range(1, len(nums)):
            for t in range(1, total+1):
                not_take = dp[prev][t]
                take = False
                if nums[i] < t:
                    take = dp[prev][t-nums[i]] 
                dp[curr][t] = take or not_take
            dp[prev] = dp[curr] # make the prev the curr so that we can start building on the curr again
        dp = np.array(dp)
        print('final dp')
        print(dp)
        return dp[curr][total]


nums = [1,2,3,4]
sol = Solution()
print(nums)
print(sol.canPartition(nums))
print('calls', sol.calls)

nums=[14,9,8,4,3,2]
sol = Solution()
print(nums)
print(sol.canPartition(nums))
print('calls', sol.calls)

nums=[1,2,5]
sol = Solution()
print(nums)
print(sol.canPartition(nums))
print('calls', sol.calls)