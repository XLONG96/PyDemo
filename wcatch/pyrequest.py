'''
Created on 2018年5月26日

@author: Administrator
'''
import requests

'''
params = {'text1':'zhimakaimen'}
r = requests.post('http://teamxlc.sinaapp.com/web1/02298884f0724c04293b4d8c0178615e/index.php', data=params)
print(r.content)
'''

for id in range(1024):
    r = requests.get('http://chinalover.sinaapp.com/web11/sql.php?id=%d' %id)
    print(r.content)