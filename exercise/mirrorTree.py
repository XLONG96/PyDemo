'''
Created on 2018年4月8日

@author: Administrator
'''
from heapq import merge

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if not pHead1:
            return pHead2
        elif not pHead2:
            return pHead1

        if pHead1.val < pHead2.val:
            pHead1.next = self.Merge(pHead1.next, pHead2)
            return pHead1
        else:
            pHead2.next = self.Merge(pHead1, pHead2.next)
            return pHead2
        
node1 = ListNode(3)
node2 = ListNode(4)
node3 = ListNode(1)
node4 = ListNode(8)

node1.next = node2
node3.next = node4

s = Solution()
node = s.Merge(node1, node3)

while node:
    print(node.val, end=" ")
    node = node.next
