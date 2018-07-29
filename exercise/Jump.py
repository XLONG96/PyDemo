'''
Created on 2018年4月14日

@author: Administrator
'''
class Solution:
    def jumpFloor(self, number):
        # write code here
        if number == 1 or number == 0:
            return 1
        return self.jumpFloor(number - 1) + self.jumpFloor(number - 2)
    
number = 100
s = Solution()

print(s.jumpFloor(number))