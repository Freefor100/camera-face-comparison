"""构建线程安全的人脸原型库，并提供不可变的候选检索快照。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

import numpy as np

from .domain import FaceSample, Person
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
    names_by_id: Mapping[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """一次性建立只读姓名映射，避免每个查询重新遍历人员。"""
        object.__setattr__(
            self, "names_by_id", MappingProxyType(dict(zip(self.person_ids, self.display_names)))
        )

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
        return apply_ranked_open_set_policy(self.rank_candidates(query_embedding), policy)

    def rank_candidates(self, query_embedding: np.ndarray) -> list[tuple[str, float]]:
        """用矩阵乘法得到第一、第二候选，不执行开放集接收判定。

        参数：
            query_embedding：待识别的人脸特征向量。
        返回：
            按分数从高到低排列的最多两个候选身份及其余弦相似度。
        前置条件：
            快照矩阵中的每一行都是同维度的 L2 单位向量。
        """

        if not self.person_ids:
            return []

        query = normalize_embedding(query_embedding)
        if self.prototype_matrix.ndim != 2 or self.prototype_matrix.shape[0] != len(
            self.person_ids
        ):
            raise ValueError("face library prototype matrix shape is invalid")
        if self.prototype_matrix.shape[1] != query.size:
            raise ValueError("query embedding dimension does not match face library")

        # 原型和查询均已 L2 归一化，矩阵点积即可一次得到所有身份的余弦相似度。
        scores = self.prototype_matrix @ query
        top_index = int(np.argmax(scores))
        ranked: list[tuple[str, float]] = [(self.person_ids[top_index], float(scores[top_index]))]
        if len(self.person_ids) > 1:
            # 屏蔽第一名后再次取最大值，避免完整排序整个人员矩阵。
            second_scores = scores.copy()
            second_scores[top_index] = -np.inf
            second_index = int(np.argmax(second_scores))
            ranked.append((self.person_ids[second_index], float(scores[second_index])))
        return ranked


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
        """用一次数据库一致读取构建原型矩阵，不执行逐人查询。"""
        people_rows, sample_rows, _, _ = repository.read_library_rows()
        return cls.from_samples(
            [repository._person_from_row(row) for row in people_rows],
            [repository._sample_from_row(row) for row in sample_rows],
        )

    @classmethod
    def from_samples(cls, people: Sequence[Person], samples: Sequence[FaceSample]):
        """从已经检查的同源记录构建矩阵，不再读取数据库。"""
        return cls(_snapshot_from_entries(_entries_from_samples(people, samples), revision=0))

    def snapshot(self) -> FaceLibrarySnapshot:
        """取得只读矩阵及其版本，整个识别任务持有同一个快照。"""
        with self._lock:
            return self._snapshot

    def rebuild(self, repository: FaceRepository) -> None:
        """一致读取后构建新矩阵，并以复制替换方式发布。"""
        self.publish(self.from_repository(repository).snapshot())

    def publish(self, replacement: FaceLibrarySnapshot) -> None:
        """发布已构建矩阵并增加版本，不修改任何旧快照。"""
        with self._lock:
            self._snapshot = FaceLibrarySnapshot(
                replacement.person_ids,
                replacement.display_names,
                replacement.prototype_matrix,
                self._snapshot.revision + 1,
            )

    def refresh_person(self, repository: FaceRepository, person_id: str) -> None:
        """一次读取后只重算指定人员行，避免组合不同提交时刻的数据。"""
        people_rows, sample_rows, _, _ = repository.read_library_rows()
        people = [repository._person_from_row(row) for row in people_rows]
        samples = [repository._sample_from_row(row) for row in sample_rows]
        replacement = _entries_from_samples(
            [person for person in people if person.id == person_id],
            [sample for sample in samples if sample.person_id == person_id],
        )
        with self._lock:
            entries = [
                (pid, name, self._snapshot.prototype_matrix[index].copy())
                for index, (pid, name) in enumerate(
                    zip(self._snapshot.person_ids, self._snapshot.display_names)
                )
                if pid != person_id
            ]
            entries.extend(replacement)
            self._snapshot = _snapshot_from_entries(entries, revision=self._snapshot.revision + 1)

    def clear(self) -> None:
        """检查失败时发布空库，不修改已交给任务的旧快照。"""
        with self._lock:
            self._snapshot = _snapshot_from_entries((), revision=self._snapshot.revision + 1)


def _entries_from_samples(people, samples):
    """按人员分组全部样本，拒绝无样本身份和无效平均向量。"""
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.person_id, []).append(normalize_embedding(sample.embedding))
    entries = []
    for person in sorted(people, key=lambda item: item.id):
        if not grouped.get(person.id):
            raise ValueError(f"人员没有有效样本：{person.display_name}")
        prototype = normalize_embedding(np.mean(grouped[person.id], axis=0))
        entries.append((person.id, person.display_name, prototype))
    return entries


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
    # 将共享矩阵标记为只读，防止常规调用代码意外原地修改快照内容。
    matrix.setflags(write=False)
    return FaceLibrarySnapshot(person_ids, display_names, matrix, revision)
