"""
Problem statement: https://neetcode.io/problems/rotate-matrix
"""
from typing import List

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        for toprow in range(len(matrix) // 2):
            # make a mirror along the forward diagonal
            for j in range(toprow, len(matrix)-toprow):
                # swapping top with right
                curr_row, curr_col = toprow, j
                mirrori, mirrorj = len(matrix) - 1 - toprow - j, len(matrix) - 1 - toprow
                matrix[curr_row][curr_col], matrix[mirrori][mirrorj] = matrix[mirrori][mirrorj], matrix[curr_row][curr_col]
                print(matrix)
                
                # swapping bottom with left
                if j < len(matrix)-1:
                    curr_row, curr_col = len(matrix) - 1 - toprow, j
                    mirrori, mirrorj = len(matrix) - 1 - toprow - j, toprow
                    print(curr_row, curr_col, mirrori, mirrorj)
                    matrix[curr_row][curr_col], matrix[mirrori][mirrorj] = matrix[mirrori][mirrorj], matrix[curr_row][curr_col]

            # now swap the top and bottom row
            matrix[toprow], matrix[len(matrix) - 1 - toprow] = matrix[len(matrix) - 1 - toprow], matrix[toprow]


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        # transpose the matrix
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if j > i:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        print(matrix)
        for col in range(len(matrix)//2):
            for row in range(len(matrix)):
                # now swap the left and right row
                matrix[row][col], matrix[row][len(matrix) - 1 - col] = matrix[row][len(matrix) - 1 - col], matrix[row][col]


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        # transpose the matrix
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if j > i:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            # since the row is done we can do the swapping for this row
            row = i
            for col in range(len(matrix)//2):
                matrix[row][col], matrix[row][len(matrix) - 1 - col] = matrix[row][len(matrix) - 1 - col], matrix[row][col]


matrix = [
  [1,2],
  [3,4]
]
sol = Solution()
print(matrix)
print('#'*30)
sol.rotate(matrix)
print(matrix)
print('#'*30)

matrix = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]
sol = Solution()
print(matrix)
print('#'*30)
sol.rotate(matrix)
print('answer')
print(matrix)
print('#'*30)