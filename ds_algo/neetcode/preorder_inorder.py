"""
Problem statement: https://neetcode.io/problems/binary-tree-from-preorder-and-inorder-traversal
"""

from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(
            self, preorder: List[int],
            inorder: List[int]
        ) -> Optional[TreeNode]:
        if len(preorder) == 0:
            return None
        else:
            root = preorder[0]
            left_preorder = []
            right_preorder = []

            left_inorder = []
            left_inorder_map = {}
            right_inorder = []
            right_inorder_map = {}
            
            found = False
            for elem in inorder:
                if elem == root:
                    found = True
                else:
                    if found is True:
                        right_inorder.append(elem)
                        right_inorder_map[elem] = True
                    else:
                        left_inorder.append(elem)
                        left_inorder_map[elem] = True
            for elem in preorder:
                if elem == root:
                    pass
                elif elem in left_inorder_map:
                    left_preorder.append(elem)
                else:
                    right_preorder.append(elem)
            root = TreeNode(root)
            root.left = self.buildTree(left_preorder, left_inorder)
            root.right = self.buildTree(right_preorder, right_inorder)
            return root
                    