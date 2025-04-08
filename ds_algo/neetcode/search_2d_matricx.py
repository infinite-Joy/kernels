"""
problem statement: https://neetcode.io/problems/search-2d-matrix
"""

from typing import List

class Solution:
    def search_row(self, matrix, l, h, target):
        while l <= h:
            mid = (l + h) // 2
            if matrix[mid][0] <= target and matrix[mid][-1] >= target:
                return mid
            elif matrix[mid][-1] < target:
                l = mid + 1
            elif matrix[mid][0] > target:
                h = mid - 1

    def search_in_row(self, row, l, h, target):
        """
        Once a suitable row is found then search within the row
        """
        while l <= h:
            mid = (l+h) // 2
            if row[mid] == target:
                return True
            elif row[mid] < target:
                l = mid + 1
            else:
                h = mid - 1
        return False

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        two binary search so logm + logn
        """
        row = self.search_row(matrix, 0, len(matrix)-1, target)
        if row is None:
            return False
        return self.search_in_row(matrix[row], 0, len(matrix[row])-1, target)
        
matrix=[[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target=3
sol = Solution()
print(sol.searchMatrix(matrix, target))