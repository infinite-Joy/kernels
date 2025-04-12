"""
course schedule II: https://neetcode.io/problems/course-schedule-ii
"""
from typing import List

class Solution:
    def dfs(self, g, node, visited, pathvisited, topsort):
        print(node, visited, pathvisited, topsort)
        visited[node] = True
        pathvisited[node] = True
        children = g[node]
        for child in children:
            if pathvisited.get(child) is True:
                return True
            if child not in visited:
                if self.dfs(g, child, visited, pathvisited, topsort):
                    return True
        topsort.append(node)
        pathvisited[node] = False

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        g = {k: [] for k in range(numCourses)}
        for course, pre in prerequisites:
            g[course].append(pre)

        print(g)

        visited = {}
        pathvisited = {}
        topsort = []
        for course in range(numCourses):
            if course not in visited:
                if self.dfs(g, course, visited, pathvisited, topsort):
                    print('course, pathvisited', course, pathvisited)
                    print('starting', course)
                    return []
        return topsort

numCourses = 3
prerequisites = [[1,0]]
sol = Solution()
print(sol.findOrder(numCourses, prerequisites))