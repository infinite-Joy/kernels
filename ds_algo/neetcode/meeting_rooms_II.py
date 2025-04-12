"""
https://neetcode.io/problems/meeting-schedule-ii
"""

"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

"""
heap idea also does not work and is too complicated.

so what we are doing is keep start times and end time in a separate array and then sort the arrays

if start time less than end time then increment the array and then increase the count. keep track of max count.
if starttime greater than array or same as array then decrement the array

"""

from typing import List
from heapq import heapify

class Solution:
    def minMeetingRooms(self, intervals: List[int]) -> int:
        maxdays = 0
        days = 0
        starttimes = sorted([i[0] for i in intervals])
        endtimes = sorted([i[1] for i in intervals])
        s, e = 0, 0
        while s < len(starttimes) and e < len(endtimes):
            if starttimes[s] < endtimes[e]:
                s += 1
                days += 1
                maxdays = max(days, maxdays)
            elif starttimes[s] == endtimes[e]:
                s += 1
                e += 1
                days = max(days-1, 0)
            else:
                e += 1
                days = max(days-1, 0)
        # while s < len(starttimes):
        #     days += len(starttimes) - s
        #     maxdays = max(days, maxdays)
        return maxdays

intervals=[(1,5),(2,6),(3,7),(4,8),(5,9)]
sol = Solution()
print(sol.minMeetingRooms(intervals))