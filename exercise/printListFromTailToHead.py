'''
Created on 2018年4月7日

@author: Administrator
'''


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        s = []
        
        while listNode is not None:
            s.append(listNode.val)
            listNode = listNode.next
        
        return s[::-1]
        
s = Solution()
node = ListNode(1)
node.next = ListNode(0)

print(s.printListFromTailToHead(node))

