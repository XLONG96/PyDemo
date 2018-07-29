'''
Created on 2018年5月17日

@author: Administrator
'''
import sys  
  
from PyQt5 import QtWidgets, QtCore, QtGui
from qt.form import Ui_Form

class mywindow(QtWidgets.QWidget, Ui_Form):    
    str1 = ''
    str2 = ''
    
    def __init__(self):    
        super(mywindow, self).__init__()    
        self.setupUi(self)

    #定义槽函数
    def add(self):
        self.str1 = self.lineEdit.text()
        self.str2 = self.lineEdit_2.text()
        
    def equal(self):
        self.result.setText(str(float(self.str1) + float(self.str2)))

app = QtWidgets.QApplication(sys.argv)
window = mywindow()
window.show()
sys.exit(app.exec_())
