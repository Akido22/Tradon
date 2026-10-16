"""GUI entrypoint. qasync merges asyncio into Qt's event loop so the same
async DataProvider/SignalEngine code the CLI uses runs here unchanged.
"""
from __future__ import annotations

import sys

import qasync
from PySide6.QtWidgets import QApplication

from tradon.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    qasync.asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
