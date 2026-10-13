"""后台解码有界缩略图，缓存不持有标准库原始大图片。"""

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QImageReader


class ThumbnailJob(QRunnable):
    """只在后台读取 QImage，不创建控件或 QPixmap。"""

    def __init__(self, owner, key, generation, sample_id):
        """保存缓存键和选中版本，迟到结果仍能被识别。"""
        super().__init__()
        self.owner, self.key = owner, key
        self.generation, self.sample_id = generation, sample_id

    def run(self):
        """按原比例解码至最长 256 像素，然后交回界面线程。"""
        reader = QImageReader(self.key[0])
        size = reader.size()
        if size.isValid():
            reader.setScaledSize(size.scaled(QSize(256, 256), Qt.KeepAspectRatio))
        image = reader.read()
        if not self.owner.closed:
            self.owner.decoded.emit(self.generation, self.sample_id, self.key, image)


class ThumbnailLoader(QObject):
    """最多缓存 64 张缩略图，最多两项同时解码。"""

    ready = Signal(int, str, object)
    decoded = Signal(int, str, object, object)

    def __init__(self, parent=None):
        """缓存和界面资源只在主线程更新。"""
        super().__init__(parent)
        self.cache = OrderedDict()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.closed = False
        self.decoded.connect(self._accept)

    def request(self, path: Path, sample_id: str, generation: int):
        """由文件时间和大小区分版本，缓存命中不再解码。"""
        if self.closed:
            return
        try:
            stat = path.stat()
        except OSError:
            self.ready.emit(generation, sample_id, QImage())
            return
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.cache:
            image = self.cache[key]
            self.cache.move_to_end(key)
            self.ready.emit(generation, sample_id, image)
        else:
            self.pool.start(ThumbnailJob(self, key, generation, sample_id))

    def _accept(self, generation, sample_id, key, image):
        """主线程更新有界缓存，页面自行检查选择版本。"""
        if self.closed:
            return
        self.cache[key] = image
        self.cache.move_to_end(key)
        while len(self.cache) > 64:
            self.cache.popitem(last=False)
        self.ready.emit(generation, sample_id, image)

    @property
    def idle(self):
        """所有已启动解码均已退出时，窗口才可释放私有线程池。"""
        return self.pool.activeThreadCount() == 0

    def cancel(self):
        """丢弃尚未启动的任务，运行中的解码结束后不再更新页面。"""
        self.closed = True
        self.pool.clear()
