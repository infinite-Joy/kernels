from typing import List
import time


class Solution:
    """
    we can do a binary search 
    on the sorted list. this will be nlogn solution
    """
    def bs(self, numbers, s, e, target):
        # time.sleep(1)
        # print(s,e)
        if s == e:
            if numbers[s] == target:
                return s
            else:
                return None
        mid = (s+e)//2
        if numbers[mid] == target:
            return mid
        if numbers[mid] < target:
            return self.bs(numbers, mid+1, e, target)
        else:
            return self.bs(numbers, s, mid-1, target)

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        e = len(numbers)-1
        for i in range(e):
            left = numbers[i]
            rem = target - left
            right = self.bs(numbers, i+1, e, rem)
            if right is not None:
                return list([i+1, right+1])

class Solution:
    """
    using a two sum approach. this can be solved using two pointer.
    complexity is O(n)
    """
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while left < right:
            sum = numbers[left] + numbers[right]
            if sum == target:
                return list([left+1, right+1])
            elif sum < target:
                left += 1
            else:
                right -= 1


numbers=[2,3,4]
target=6
sol = Solution()
print(sol.twoSum(numbers, target))