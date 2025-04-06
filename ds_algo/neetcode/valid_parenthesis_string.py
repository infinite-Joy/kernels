"""
https://neetcode.io/problems/valid-parenthesis-string

because recursion there is also a stack so we can run a dp based recursion on it.
or maybe just backtracking

go through all the paths that will end in whole stack being finished.
we need double stack. one to maintain all the paths
if any one of the paths is over then its true
"""


class Solution:
    def check_valid_string_rec(self, s, i, stack, replaced=None):
        if i >= len(s):
            return stack == 0
        curr_val = replaced or s[i]
        if curr_val == '(':
            stack += 1
            return self.check_valid_string_rec(s, i+1, stack)
        elif curr_val == ')':
            stack = max(0, stack-1)
            return self.check_valid_string_rec(s, i+1, stack)
        else:
            return (
                self.check_valid_string_rec(s, i, stack, replaced='(') or
                self.check_valid_string_rec(s, i, stack, replaced=')') or
                self.check_valid_string_rec(s, i+1, stack, replaced=None)
            )

        

    def checkValidString(self, s: str) -> bool:
        return self.check_valid_string_rec(s, 0, 0)


# greedy solutuon. the above backtracking is 3**n solution

class Solution:
    def checkValidString(self, s: str) -> bool:
        leftmax, leftmin = 0, 0
        for elem in s:
            if elem == '(':
                leftmin += 1
                leftmax += 1
            elif elem == ')':
                leftmin -= 1
                leftmax -= 1
            else:
                leftmin -= 1
                leftmax += 1
            # do the check before running the operation
            # this is because of some case like ))
            if leftmax < 0:
                return False
            if leftmin < 0:
                leftmin = 0

        return leftmin == 0


s = "((**)"
sol = Solution()
print(sol.checkValidString(s))

s = "(((*)"
sol = Solution()
print(sol.checkValidString(s))

s = "))"
sol = Solution()
print(sol.checkValidString(s))