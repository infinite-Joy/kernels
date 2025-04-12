from typing import List
import time

class Solution:
    count = 0
    def wellformed(self, s):
        stack = []
        for item in s:
            if item == '(':
                stack.append(item)
            elif item == ')' and stack and stack[-1] == '(':
                stack.pop()
            else:
                return False
        return len(stack) == 0
    
    def backtrack(self, n, s):
        self.count += 1
        # print(s)
        # time.sleep(1)
        if len(s)>=2*n:
            if self.wellformed(s):
                yield "".join(s)
            else:
                yield from [] # nothing
        else:
            s.append('(')
            yield from self.backtrack(n, s) # not closing
            s.pop() # remove the startimg
            s.append(')')
            yield from self.backtrack(n, s) # not closing
            s.pop() # remove the startimg

    def generateParenthesis(self, n: int) -> List[str]:
        s = []
        return list(self.backtrack(n, s))


class Solution2:
    count = 0
    def wellformed(self, s):
        stack = []
        for item in s:
            if item == '(':
                stack.append(item)
            elif item == ')' and stack and stack[-1] == '(':
                stack.pop()
            else:
                return False
        return len(stack) == 0
    
    def backtrack(self, n, s, copen, cclose):
        self.count += 1
        if len(s)>=2*n:
            if self.wellformed(s):
                yield "".join(s)
            else:
                yield from [] # nothing
        else:
            if copen < n:
                s.append('(')
                yield from self.backtrack(n, s, copen+1, cclose) # not closing
                s.pop() # remove the startimg
                if copen > cclose:
                    s.append(')')
                    yield from self.backtrack(n, s, copen, cclose+1)
                    s.pop()


    def generateParenthesis(self, n: int) -> List[str]:
        s = []
        return list(self.backtrack(n, s, 0, 0))


n = 3
sol = Solution()
print(sol.generateParenthesis(n))
print(sol.count)

n = 3
sol = Solution2()
print(sol.generateParenthesis(n))
print(sol.count)
# Output: ["((()))","(()())","(())()","()(())","()()()"]