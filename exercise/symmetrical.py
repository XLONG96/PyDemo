'''
Created on 2018年4月7日

@author: Administrator
'''

class TreeNode:
    def __init__(self, x):         
        self.val = x         
        self.left = None
        self.right = None

class Solution:
    def isSymmetrical(self, pRoot):
        return self.judge(pRoot, pRoot)
    
    def judge(self, node1, node2):
        if node1 is not None and node2 is not None and node1.val == node2.val:
            return self.judge(node1.left, node2.right) \
                and self.judge(node1.right, node2.left)
        return node1 is None and node2 is None

solution = Solution()

node = TreeNode(None)

print(solution.isSymmetrical(node))



