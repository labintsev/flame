import sys
import controller

if __name__ == '__main__':
    app = controller.QtWidgets.QApplication(sys.argv)
    mw = controller.Widget()
    mw.show()
    app.exec()
