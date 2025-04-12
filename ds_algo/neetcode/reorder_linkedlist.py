"""
problem: https://neetcode.io/problems/reorder-linked-list
"""


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class ParentNode:
    def __init__(self, val)
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        find the middle position
        from the middle reverse the linkedlist
        then merge together

        0   1   2   3   4   5   6
                    i
                                j
        """
        # find the middle position
        # using fast and slow pointer
        slow = head
        fast = head
        while fast.next:
            fast = fast.next
            if fast:
                fast = fast.next
            slow = slow.next
        middle = slow

        # reverse
        prev = None
        cnode = middle
        nnode = cnode.next
        while nnode:
            cnode.next = prev
            prev = cnode
            cnode = nnode
            nnode = cnode.next
        lastnode = cnode

        # now join
        curr = head
        normal = head
        reverse = lastnode
        while curr != middle:
            normal_next = normal.next
            reverse_next = reverse.next
            curr = normal
            curr.next = reverse