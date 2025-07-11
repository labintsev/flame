import sys
import gui

if __name__ == '__main__':
    app = gui.QtWidgets.QApplication(sys.argv)
    mw = gui.Widget()
    mw.show()
    app.exec()
