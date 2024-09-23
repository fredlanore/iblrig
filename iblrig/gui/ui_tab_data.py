# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'iblrig\gui\ui_tab_data.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TabData(object):
    def setupUi(self, TabData):
        TabData.setObjectName("TabData")
        TabData.resize(776, 678)
        self.verticalLayout = QtWidgets.QVBoxLayout(TabData)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableView = QtWidgets.QTableView(TabData)
        self.tableView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableView.setTabKeyNavigation(False)
        self.tableView.setProperty("showDropIndicator", False)
        self.tableView.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tableView.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableView.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.tableView.setShowGrid(False)
        self.tableView.setSortingEnabled(True)
        self.tableView.setWordWrap(False)
        self.tableView.setObjectName("tableView")
        self.tableView.horizontalHeader().setCascadingSectionResizes(True)
        self.tableView.horizontalHeader().setHighlightSections(False)
        self.tableView.verticalHeader().setVisible(False)
        self.verticalLayout.addWidget(self.tableView)
        self.horizontalWidget = QtWidgets.QWidget(TabData)
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEditFilter = QtWidgets.QLineEdit(self.horizontalWidget)
        self.lineEditFilter.setObjectName("lineEditFilter")
        self.horizontalLayout.addWidget(self.lineEditFilter)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButtonUpdate = QtWidgets.QPushButton(self.horizontalWidget)
        self.pushButtonUpdate.setObjectName("pushButtonUpdate")
        self.horizontalLayout.addWidget(self.pushButtonUpdate)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout.addWidget(self.horizontalWidget)

        self.retranslateUi(TabData)
        QtCore.QMetaObject.connectSlotsByName(TabData)

    def retranslateUi(self, TabData):
        _translate = QtCore.QCoreApplication.translate
        TabData.setWindowTitle(_translate("TabData", "Form"))
        self.tableView.setStatusTip(_translate("TabData", "Double-click a row to open the respective folder"))
        self.lineEditFilter.setStatusTip(_translate("TabData", "Filter table by subject"))
        self.lineEditFilter.setPlaceholderText(_translate("TabData", "Filter by Subject"))
        self.pushButtonUpdate.setStatusTip(_translate("TabData", "Update table"))
        self.pushButtonUpdate.setText(_translate("TabData", "Update"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TabData = QtWidgets.QWidget()
    ui = Ui_TabData()
    ui.setupUi(TabData)
    TabData.show()
    sys.exit(app.exec_())