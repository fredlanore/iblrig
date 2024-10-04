import argparse
import logging
import subprocess
import sys
import traceback
from collections.abc import Callable
from inspect import signature
from pathlib import Path
from shutil import disk_usage
from typing import Any

from PyQt5 import QtGui
from PyQt5.QtCore import (
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QAction, QLineEdit, QListView, QProgressBar, QPushButton
from requests import HTTPError

from iblrig.constants import BASE_PATH
from iblrig.gui import resources_rc  # noqa: F401
from iblrig.net import get_remote_devices
from iblrig.pydantic_definitions import RigSettings
from iblutil.util import dir_size
from one.webclient import AlyxClient

log = logging.getLogger(__name__)


def convert_uis():
    """A wrapper for PyQt5's pyuic5 and pyrcc5, set up for development on iblrig."""
    parser = argparse.ArgumentParser()
    parser.add_argument('pattern', nargs='?', default='*.*', type=str)
    args = parser.parse_args()

    gui_path = BASE_PATH.joinpath('iblrig', 'gui')
    files = set([f for f in gui_path.glob(args.pattern)])

    for filename_in in files.intersection(gui_path.glob('*.qrc')):
        rel_path_in = filename_in.relative_to(BASE_PATH)
        rel_path_out = rel_path_in.with_stem(rel_path_in.stem + '_rc').with_suffix('.py')
        args = ['pyrcc5', str(rel_path_in), '-o', str(rel_path_out)]
        print(' '.join(args))
        subprocess.check_output(args, cwd=BASE_PATH)

    for filename_in in files.intersection(gui_path.glob('*.ui')):
        rel_path_in = filename_in.relative_to(BASE_PATH)
        rel_path_out = rel_path_in.with_suffix('.py')
        args = ['pyuic5', str(rel_path_in), '-o', str(rel_path_out), '-x', '--import-from=iblrig.gui']
        print(' '.join(args))
        subprocess.check_output(args, cwd=BASE_PATH)


class WorkerSignals(QObject):
    """
    Signals used by the Worker class to communicate with the main thread.

    Attributes
    ----------
    finished : pyqtSignal
        Signal emitted when the worker has finished its task.

    error : pyqtSignal(tuple)
        Signal emitted when an error occurs. The signal carries a tuple with the exception type,
        exception value, and the formatted traceback.

    result : pyqtSignal(Any)
        Signal emitted when the worker has successfully completed its task. The signal carries
        the result of the task.

    progress : pyqtSignal(int)
        Signal emitted to report progress during the task. The signal carries an integer value.
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class DiskSpaceIndicator(QProgressBar):
    """A custom progress bar widget that indicates the disk space usage of a specified directory."""

    def __init__(self, *args, directory: Path | None, percent_threshold: int = 90, **kwargs):
        """
        Initialize the DiskSpaceIndicator with the specified directory and threshold percentage.

        Parameters
        ----------
        *args : tuple
            Variable length argument list (passed to QProgressBar).
        directory : Path or None
            The directory path to monitor for disk space usage.
        percent_threshold : int, optional
            The threshold percentage at which the progress bar changes color to red. Default is 90.
        **kwargs : dict
            Arbitrary keyword arguments (passed to QProgressBar).
        """
        super().__init__(*args, **kwargs)
        self._directory = directory
        self._percent_threshold = percent_threshold
        self._percent_full = float('nan')
        self.setEnabled(False)
        if self._directory is not None:
            self.update_data()

    def update_data(self):
        """Update the disk space information."""
        worker = Worker(self._get_size)
        worker.signals.result.connect(self._on_get_size_result)
        QThreadPool.globalInstance().start(worker)

    @property
    def critical(self) -> bool:
        """True if the disk space usage is above the given threshold percentage."""
        return self._percent_full > self._percent_threshold

    def _get_size(self):
        """Get the disk usage information for the specified directory."""
        usage = disk_usage(self._directory.anchor)
        self._percent_full = usage.used / usage.total * 100
        self._gigs_dir = dir_size(self._directory) / 1024**3
        self._gigs_free = usage.free / 1024**3

    def _on_get_size_result(self, result):
        """Handle the result of getting disk usage information and update the progress bar accordingly."""
        self.setEnabled(True)
        self.setValue(round(self._percent_full))
        if self.critical:
            p = self.palette()
            p.setColor(QtGui.QPalette.Highlight, QtGui.QColor('red'))
            self.setPalette(p)
        self.setStatusTip(f'{self._directory}: {self._gigs_dir:.1f} GB  •  ' f'available space: {self._gigs_free:.1f} GB')


class Worker(QRunnable):
    """
    A generic worker class for executing functions concurrently in a separate thread.

    This class is designed to run functions concurrently in a separate thread and emit signals
    to communicate the results or errors back to the main thread.

    Adapted from: https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/

    Attributes
    ----------
    fn : Callable
        The function to be executed concurrently.

    args : tuple
        Positional arguments for the function.

    kwargs : dict
        Keyword arguments for the function.

    signals : WorkerSignals
        An instance of WorkerSignals used to emit signals.

    Methods
    -------
    run() -> None
        The main entry point for running the worker. Executes the provided function and
        emits signals accordingly.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        """
        Initialize the Worker instance.

        Parameters
        ----------
        fn : Callable
            The function to be executed concurrently.

        *args : tuple
            Positional arguments for the function.

        **kwargs : dict
            Keyword arguments for the function.
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals: WorkerSignals = WorkerSignals()
        if 'progress_callback' in signature(fn).parameters:
            self.kwargs['progress_callback'] = self.signals.progress

    def run(self) -> None:
        """
        Execute the provided function and emit signals accordingly.

        This method is the main entry point for running the worker. It executes the provided
        function and emits signals to communicate the results or errors back to the main thread.

        Returns
        -------
        None
        """
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:  # noqa: E722
            # Handle exceptions and emit error signal with exception details
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Emit result signal with the result of the task
            self.signals.result.emit(result)
        finally:
            # Emit the finished signal to indicate completion
            self.signals.finished.emit()


class RemoteDevicesListView(QListView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)  # needed for status tips

    def getDevices(self):
        out = []
        for idx in self.selectedIndexes():
            out.append(self.model().itemData(idx)[Qt.UserRole])
        return out


class RemoteDevicesItemModel(QStandardItemModel):
    def __init__(self, *args, iblrig_settings: RigSettings, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote_devices = get_remote_devices(iblrig_settings=iblrig_settings)
        self.update()

    @pyqtSlot()
    def update(self):
        self.clear()
        for device_name, device_address in self.remote_devices.items():
            item = QStandardItem(device_name)
            item.setStatusTip(f'Remote Device "{device_name}" - {device_address}')
            item.setData(device_name, Qt.UserRole)
            self.appendRow(item)


class AlyxObject(QObject):
    """
    A class to manage user authentication with an AlyxClient.

    This class provides methods to log in and log out users, emitting signals to indicate changes in authentication status.

    Parameters
    ----------
    alyxUrl : str, optional
        The base URL for the Alyx API. If provided, an AlyxClient will be created.
    alyxClient : AlyxClient, optional
        An existing AlyxClient instance. If provided, it will be used for authentication.

    Attributes
    ----------
    isLoggedIn : bool
        Indicates whether a user is currently logged in.
    username : str or None
        The username of the logged-in user, or None if not logged in.
    statusChanged : pyqtSignal
        Emitted when the login status changes (logged in or out). The signal carries a boolean indicating the new status.
    loggedIn : pyqtSignal
        Emitted when a user logs in. The signal carries a string representing the username.
    loggedOut : pyqtSignal
        Emitted when a user logs out. The signal carries a string representing the username.
    loginFailed : pyqtSignal
        Emitted when a login attempt fails. The signal carries a string representing the username.
    """

    statusChanged = pyqtSignal(bool)
    loggedIn = pyqtSignal(str)
    loggedOut = pyqtSignal(str)
    loginFailed = pyqtSignal(str)

    def __init__(self, *args, alyxUrl: str | None = None, alyxClient: AlyxClient | None = None, **kwargs):
        """
        Initializes the AlyxObject.

        Parameters
        ----------
        *args : tuple
            Positional arguments for QObject.
        alyxUrl : str, optional
            The base URL for the Alyx API.
        alyxClient : AlyxClient, optional
            An existing AlyxClient instance.
        **kwargs : dict
            Keyword arguments for QObject.
        """
        super().__init__(*args, **kwargs)
        self._icon = super().icon()

        if alyxUrl is not None:
            self.client = AlyxClient(base_url=alyxUrl, silent=True)
        else:
            self.client = alyxClient

    @pyqtSlot(str)
    @pyqtSlot(str, str)
    @pyqtSlot(str, str, bool)
    def logIn(self, username: str, password: str | None = None, cacheToken: bool = False) -> bool:
        """
        Logs in a user with the provided username and password.

        Emits the loggedIn and statusChanged signals if the logout is successful, and the loginFailed signal otherwise.

        Parameters
        ----------
        username : str
            The username of the user attempting to log in.
        password : str or None, optional
            The password of the user. If None, the login will proceed without a password.
        cacheToken : bool, optional
            Whether to cache the authentication token.

        Returns
        -------
        bool
            True if the login was successful, False otherwise.
        """
        if self.client is None:
            return False
        try:
            self.client.authenticate(username, password, cache_token=cacheToken, force=password is not None)
        except HTTPError as e:
            if e.errno == 400 and any(x in e.response.text for x in ('credentials', 'required')):
                log.error(e.filename)
                self.loginFailed.emit(username)
            else:
                raise e
        if status := self.client.is_logged_in and self.client.user == username:
            log.debug(f"Logged into {self.client.base_url} as user '{username}'")
            self.statusChanged.emit(True)
            self.loggedIn.emit(username)
        return status

    @pyqtSlot()
    def logOut(self) -> None:
        """
        Logs out the currently logged-in user.

        Emits the loggedOut and statusChanged signals if the logout is successful.
        """
        if self.client is None or not self.isLoggedIn:
            return
        username = self.client.user
        self.client.logout()
        if not (connected := self.client.is_logged_in):
            log.debug(f"User '{username}' logged out of {self.client.base_url}")
            self.statusChanged.emit(connected)
            self.loggedOut.emit(username)

    @property
    def isLoggedIn(self):
        """Indicates whether a user is currently logged in."""
        return self.client.is_logged_in if isinstance(self.client, AlyxClient) else False

    @property
    def username(self) -> str | None:
        """The username of the logged-in user, or None if not logged in."""
        return self.client.user if self.isLoggedIn else None


class LineEditAlyxUser(QLineEdit):
    """
    A custom QLineEdit widget for managing user login with an AlyxObject.

    This widget displays a checkmark icon to indicate the connection status
    and allows the user to input their username for logging in.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the QLineEdit constructor.
    alyx : AlyxObject
        An instance of AlyxObject used to manage login and connection status.
    **kwargs : dict
        Keyword arguments passed to the QLineEdit constructor.
    """

    def __init__(self, *args, alyx: AlyxObject, **kwargs):
        """
        Initializes the LineEditAlyxUser widget.

        Sets up the checkmark icon, connects signals for login status,
        and configures the line edit based on the AlyxObject's state.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the QLineEdit constructor.
        alyx : AlyxObject
            An instance of AlyxObject.
        **kwargs : dict
            Keyword arguments passed to the QLineEdit constructor.
        """
        super().__init__(*args, **kwargs)
        self.alyx = alyx

        # Use a QAction to indicate the connection status
        self._checkmarkIcon = QAction(parent=self, icon=QtGui.QIcon(':/images/check'))
        self.addAction(self._checkmarkIcon, self.ActionPosition.TrailingPosition)

        if self.alyx.client is None:
            self.setEnabled(False)
        else:
            self.setPlaceholderText('not logged in')
            self.alyx.statusChanged.connect(self._onStatusChanged)
            self.returnPressed.connect(self.logIn)
            self._onStatusChanged(self.alyx.isLoggedIn)

    @pyqtSlot(bool)
    def _onStatusChanged(self, connected: bool):
        """Set some of the widget's properties depending on the current connection-status."""
        self._checkmarkIcon.setVisible(connected)
        self._checkmarkIcon.setToolTip(f'Connected to {self.alyx.client.base_url}' if connected else '')
        self.setText(self.alyx.username or '')
        self.setReadOnly(connected)

    @pyqtSlot()
    def logIn(self):
        """Attempt to log in using the line edit's current text."""
        self.alyx.logIn(self.text())


class StatefulButton(QPushButton):
    """
    A QPushButton that maintains an active/inactive state and emits different signals
    based on its state when clicked.

    Parameters
    ----------
    active : bool, optional
        Initial state of the button (default is False).

    Attributes
    ----------
    clickedWhileActive : pyqtSignal
        Emitted when the button is clicked while it is in the active state.
    clickedWhileInactive : pyqtSignal
        Emitted when the button is clicked while it is in the inactive state.
    stateChanged : pyqtSignal
        Emitted when the button's state has changed. The signal carries the new state.
    """

    clickedWhileActive = pyqtSignal()
    clickedWhileInactive = pyqtSignal()
    stateChanged = pyqtSignal(bool)

    def __init__(self, *args, active: bool = False, **kwargs):
        """
        Initialize the StateButton with the specified active state.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to the QPushButton constructor.
        active : bool, optional
            Initial state of the button (default is False).
        **kwargs : dict
            Keyword arguments to be passed to the QPushButton constructor.
        """
        super().__init__(*args, **kwargs)
        self._isActive = active
        self.clicked.connect(self._onClick)

    @pyqtProperty(bool)
    def isActive(self) -> bool:
        """
        Get the active state of the button.

        Returns
        -------
        bool
            True if the button is active, False otherwise.
        """
        return self._isActive

    @pyqtSlot(bool)
    def setActive(self, active: bool):
        """
        Set the active state of the button.

        Emits `stateChanged` if the state has changed.

        Parameters
        ----------
        active : bool
            The new active state of the button.
        """
        if self._isActive != active:
            self._isActive = active
            self.stateChanged.emit(self._isActive)

    @pyqtSlot()
    def _onClick(self):
        """
        Handle the button click event.

        Emits `clickedWhileActive` if the button is active,
        otherwise emits `clickedWhileInactive`.
        """
        if self._isActive:
            self.clickedWhileActive.emit()
        else:
            self.clickedWhileInactive.emit()
