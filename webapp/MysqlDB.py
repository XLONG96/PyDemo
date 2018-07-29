'''
Created on 2018年5月2日

@author: Administrator
'''

import mysql.connector as mysql
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 创建对象的基类:
Base = declarative_base()

class User(Base):
    __tablename__ = 'linuxtest'
    
    userid = Column(String(50), primary_key=True)
    mysql_version = Column(String(50))
    linux_version = Column(String(300))
    eth0_HW = Column(String(50))
    
    def __str__(self):
        return self.userid + " " + self.mysql_version + \
            " " + self.linux_version + " " + self.eth0_HW
     
# ORM方式   
def getUser():
        
    # 初始化数据库连接:
    engine = create_engine('mysql+mysqlconnector://root:@localhost:3306/test')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)
    
    session = DBSession()
    
    user = session.query(User).one()
    
    session.close()
    
    return user


# 普通方式
def getUserById(id):
    conn = mysql.connect(user='root', password='', database='test')
    cursor = conn.cursor();
    
    cursor.execute('select * from user where id=%s', (id,))
    values = cursor.fetchall()
    
    print(values)
    
    cursor.close()
    conn.close()
    
    return values


if __name__ == '__main__':
    print(getUser())





