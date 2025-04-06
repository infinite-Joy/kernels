"""
count number of islands
https://neetcode.io/problems/count-number-of-islands
"""

from collections import deque
from typing import List
import itertools

class Solution:
    def get_children(self, grid, i, j):
        directions = [[1, 0], [0, -1], [-1, 0], [0, 1]]
        for di, dj in directions:
            childi, childj = i+di, j+dj
            if 0 <= childi < len(grid) and 0 <= childj < len(grid[0]) and grid[childi][childj] == '1':
                yield childi, childj

    def bfs(self, grid, i, j, visited):
        queue = deque()
        queue.append((i, j))
        visited[(i, j)] = True
        while queue:
            nodei, nodej = queue.popleft()
            neighbours = self.get_children(grid, nodei, nodej)
            for ni, nj in neighbours:
                if (ni, nj) not in visited:
                    queue.append((ni, nj))
                    visited[(ni, nj)] = True

    def dfs(self, grid, i, j, visited):
        visited[(i, j)] = True
        neighbours = self.get_children(grid, i, j)
        for ni, nj in neighbours:
            if (ni, nj) not in visited:
                self.dfs(grid, ni, nj, visited)

    def numIslands(self, grid: List[List[str]]) -> int:
        visited = {}
        num_islands = 0
        for i, j in itertools.product(range(len(grid)), range(len(grid[0]))):
            if grid[i][j]=='1' and (i, j) not in visited:
                print('inside num islands', i, j)
                num_islands += 1
                # self.bfs(grid, i, j, visited)
                self.dfs(grid, i, j, visited)
        return num_islands


grid=[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
sol = Solution()
print(sol.numIslands(grid))

grid=[["1","1","0","0","1"],
      ["1","1","0","0","1"],
      ["0","0","1","0","0"],
      ["0","0","0","1","1"]]
sol = Solution()
print(sol.numIslands(grid))