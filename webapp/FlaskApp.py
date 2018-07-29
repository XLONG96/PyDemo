'''
Created on 2018年5月2日

@author: Administrator
'''

from flask import Flask, request, render_template
from webapp.MysqlDB import getUser

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    user = getUser()
    userid = user.userid
    mysql_version = user.mysql_version
    linux_version = user.linux_version
    eth0_HW = user.eth0_HW
    return render_template('home.html', userid=userid, mysql_version=mysql_version, \
                           linux_version=linux_version, eth0_HW=eth0_HW)


if __name__ == '__main__':
    app.run()