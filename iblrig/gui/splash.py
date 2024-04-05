import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QSequentialAnimationGroup, QThreadPool, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from typing_extensions import override

from iblrig import __version__ as version
from iblrig.constants import COPYRIGHT_YEAR
from iblrig.gui.tools import Worker
from iblrig.gui.ui_splash import Ui_splash
from iblrig.hardware_validation import Result, get_all_validators
from iblrig.path_helper import load_pydantic_yaml
from iblrig.pydantic_definitions import HardwareSettings, RigSettings


class Splash(QDialog, Ui_splash):
    validation_results: list[Result] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint)
        self.installEventFilter(self)

        # store arguments as members
        self.hardware_settings = load_pydantic_yaml(HardwareSettings)
        self.rig_settings = load_pydantic_yaml(RigSettings)

        # update a few strings
        self.labelVersion.setText(f'v{version}')
        self.labelCopyright.setText(f'© {COPYRIGHT_YEAR}, International Brain Laboratory')

        # extremely important animation
        self.hat.setProperty('pos', QPoint(0, -250))
        animation_1 = QPropertyAnimation(self, b'windowOpacity')
        animation_1.setStartValue(0.0)
        animation_1.setEndValue(1.0)
        animation_1.setDuration(200)
        animation_2 = QPropertyAnimation(self.hat, b'pos')
        animation_2.setEasingCurve(QEasingCurve.InQuad)
        animation_2.setEndValue(QPoint(0, 40))
        animation_2.setDuration(500)
        self.animation_group = QSequentialAnimationGroup()
        self.animation_group.addAnimation(animation_1)
        self.animation_group.addAnimation(animation_2)
        self.animation_group.start()

        # start timer for force close
        QTimer.singleShot(20000, self.stop_and_close)

        worker = Worker(self.validation)
        worker.signals.finished.connect(self.close)
        QThreadPool.globalInstance().tryStart(worker)

    def validation(self):
        for validator in get_all_validators():
            validator_instance = validator(hardware_settings=self.hardware_settings, iblrig_settings=self.rig_settings)
            self.labelStatus.setText(f'Validating {validator_instance.name} ...')
            for result in validator_instance.run():
                self.validation_results.append(result)

    def stop_and_close(self):
        self.close()

    @override
    def close(self):
        print(self.validation_results)
        super().close()

    @override
    def eventFilter(self, obj, event):
        """Disregard all key-presses"""
        return obj is self and event.type() == QtCore.QEvent.KeyPress


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a = Splash(self)
        a.exec()


def main():
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
