from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from .face_engine import normalize_embedding
from .open_set_policy import OpenSetDecision, ScoreThresholdPolicy, apply_ranked_open_set_policy
from .repository import FaceRepository


@dataclass(frozen=True)
class FaceLibrarySnapshot:
    """一次识别任务使用的、不可变的人员原型快照。"""

    person_ids: tuple[str, ...]
    display_names: tuple[str, ...]
    prototype_matrix: np.ndarray
    revision: int

    def search(
        self,
        query_embedding: np.ndarray,
        policy: ScoreThresholdPolicy,
    ) -> OpenSetDecision:
        """用矩阵乘法检索全部身份并应用最高分阈值规则。

        参数：
            query_embedding：待识别的人脸特征向量。
            policy：当前部署的最高分阈值策略。
        返回：
            第一候选、第二候选及开放集接收结果。
        前置条件：
            快照矩阵中的每一行都是同维度的 L2 单位向量。
        """

        if not isinstance(policy, ScoreThresholdPolicy):
            raise TypeError("runtime face library search requires ScoreThresholdPolicy")
        if not self.person_ids:
            return apply_ranked_open_set_policy((), policy)

        query = normalize_embedding(query_embedding)
        if self.prototype_matrix.ndim != 2 or self.prototype_matrix.shape[0] != len(
            self.person_ids
        ):
            raise ValueError("face library prototype matrix shape is invalid")
        if self.prototype_matrix.shape[1] != query.size:
            raise ValueError("query embedding dimension does not match face library")

        scores = self.prototype_matrix @ query
        top_index = int(np.argmax(scores))
        ranked: list[tuple[str, float]] = [
            (self.person_ids[top_index], float(scores[top_index]))
        ]
        if len(self.person_ids) > 1:
            second_scores = scores.copy()
            second_scores[top_index] = -np.inf
            second_index = int(np.argmax(second_scores))
            ranked.append((self.person_ids[second_index], float(scores[second_index])))
        return apply_ranked_open_set_policy(ranked, policy)


class InMemoryFaceLibrary:
    """从 SQLite 样本构建人员原型，并以复制替换方式提供并发查询。"""

    def __init__(self, snapshot: FaceLibrarySnapshot) -> None:
        """创建带初始快照的内存标准库。"""

        self._lock = RLock()
        self._snapshot = snapshot

    @classmethod
    def empty(cls) -> InMemoryFaceLibrary:
        """创建空的内存标准库，用于完整性检查尚未通过的启动阶段。"""

        return cls(_snapshot_from_entries((), revision=0))

    @classmethod
    def from_repository(cls, repository: FaceRepository) -> InMemoryFaceLibrary:
        """读取仓库中的全部样本并构建初始人员原型矩阵。"""

        person_rows = sorted(repository.list_people(), key=lambda person: person.id)
        entries = []
        for person in person_rows:
            samples = repository.list_samples(person.id)
            if not samples:
                continue
            normalized = [normalize_embedding(sample.embedding) for sample in samples]
            prototype = normalize_embedding(np.mean(normalized, axis=0))
            entries.append((person.id, person.display_name, prototype))
        return cls(_snapshot_from_entries(entries, revision=0))

    def snapshot(self) -> FaceLibrarySnapshot:
        """返回当前不可变快照，供一次识别任务持有。"""

        with self._lock:
            return self._snapshot

    def rebuild(self, repository: FaceRepository) -> None:
        """从仓库全部样本重建矩阵，并一次性替换旧快照。"""

        person_rows = sorted(repository.list_people(), key=lambda person: person.id)
        entries = []
        for person in person_rows:
            samples = repository.list_samples(person.id)
            if not samples:
                continue
            normalized = [normalize_embedding(sample.embedding) for sample in samples]
            prototype = normalize_embedding(np.mean(normalized, axis=0))
            entries.append((person.id, person.display_name, prototype))
        with self._lock:
            self._snapshot = _snapshot_from_entries(
                entries, revision=self._snapshot.revision + 1
            )

    def refresh_person(self, repository: FaceRepository, person_id: str) -> None:
        """只重新计算一个身份的原型，并复制替换包含它的新矩阵。"""

        person = repository.get_person(person_id)
        samples = [] if person is None else repository.list_samples(person_id)
        replacement = None
        if person is not None and samples:
            normalized = [normalize_embedding(sample.embedding) for sample in samples]
            replacement = (person.id, person.display_name, normalize_embedding(np.mean(normalized, axis=0)))

        with self._lock:
            entries = [
                (current_id, current_name, self._snapshot.prototype_matrix[index].copy())
                for index, (current_id, current_name) in enumerate(
                    zip(self._snapshot.person_ids, self._snapshot.display_names, strict=True)
                )
                if current_id != person_id
            ]
            if replacement is not None:
                entries.append(replacement)
            entries.sort(key=lambda entry: entry[0])
            self._snapshot = _snapshot_from_entries(
                entries, revision=self._snapshot.revision + 1
            )

    def clear(self) -> None:
        """清空内存标准库并递增版本号。"""

        with self._lock:
            self._snapshot = _snapshot_from_entries(
                (), revision=self._snapshot.revision + 1
            )


def _snapshot_from_entries(
    entries: list[tuple[str, str, np.ndarray]] | tuple[()],
    *,
    revision: int,
) -> FaceLibrarySnapshot:
    """把排序后的身份原型转换为只读连续矩阵。"""

    ordered = sorted(entries, key=lambda entry: entry[0])
    person_ids = tuple(entry[0] for entry in ordered)
    display_names = tuple(entry[1] for entry in ordered)
    if ordered:
        matrix = np.ascontiguousarray(np.stack([entry[2] for entry in ordered]), dtype=np.float32)
    else:
        matrix = np.empty((0, 0), dtype=np.float32)
    matrix.setflags(write=False)
    return FaceLibrarySnapshot(person_ids, display_names, matrix, revision)
