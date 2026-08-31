from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .domain import FaceSample, Person


@dataclass(frozen=True)
class SampleInput:
    """A sample prepared by the enrollment workflow for one SQLite transaction."""

    image_path: str
    embedding: np.ndarray
    pose: str
    quality: dict[str, float]


class FaceRepository:
    """SQLite-backed storage for people, face embeddings, and recognition logs."""

    def __init__(self, database_path: Path) -> None:
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def close(self) -> None:
        self._connection.close()

    def create_person(self, display_name: str) -> Person:
        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("display_name must not be empty")
        person = Person(
            id=str(uuid4()),
            display_name=normalized_name,
            created_at=_now(),
        )
        with self._connection:
            self._connection.execute(
                "INSERT INTO persons (id, display_name, created_at) VALUES (?, ?, ?)",
                (person.id, person.display_name, person.created_at.isoformat()),
            )
        return person

    def create_person_with_samples(
        self,
        *,
        person_id: str,
        display_name: str,
        samples: Sequence[SampleInput],
    ) -> Person:
        """Persist a new identity and all of its samples atomically."""

        normalized_name = display_name.strip()
        if not normalized_name:
            raise ValueError("display_name must not be empty")
        if not samples:
            raise ValueError("at least one sample is required")

        person = Person(id=person_id, display_name=normalized_name, created_at=_now())
        prepared_samples = [
            self._make_sample(
                person_id=person.id,
                image_path=sample.image_path,
                embedding=sample.embedding,
                pose=sample.pose,
                quality=sample.quality,
            )
            for sample in samples
        ]
        with self._connection:
            self._connection.execute(
                "INSERT INTO persons (id, display_name, created_at) VALUES (?, ?, ?)",
                (person.id, person.display_name, person.created_at.isoformat()),
            )
            for sample in prepared_samples:
                self._insert_sample(sample)
        return person

    def add_sample(
        self,
        *,
        person_id: str,
        image_path: str,
        embedding: np.ndarray,
        pose: str,
        quality: dict[str, float],
    ) -> FaceSample:
        sample = self._make_sample(
            person_id=person_id,
            image_path=image_path,
            embedding=embedding,
            pose=pose,
            quality=quality,
        )
        with self._connection:
            self._insert_sample(sample)
        return sample

    def list_people(self) -> list[Person]:
        rows = self._connection.execute(
            "SELECT id, display_name, created_at FROM persons ORDER BY created_at, display_name"
        ).fetchall()
        return [
            Person(
                id=row["id"],
                display_name=row["display_name"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_person(self, person_id: str) -> Person | None:
        row = self._connection.execute(
            "SELECT id, display_name, created_at FROM persons WHERE id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return None
        return Person(
            id=row["id"],
            display_name=row["display_name"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_samples(self, person_id: str | None = None) -> list[FaceSample]:
        query = """
            SELECT id, person_id, image_path, embedding_blob, embedding_dim,
                   pose, quality_json, created_at
            FROM face_samples
        """
        parameters: tuple[str, ...] = ()
        if person_id is not None:
            query += " WHERE person_id = ?"
            parameters = (person_id,)
        query += " ORDER BY created_at"
        rows = self._connection.execute(query, parameters).fetchall()
        return [self._sample_from_row(row) for row in rows]

    def record_recognition(
        self,
        *,
        decision: str,
        person_id: str | None,
        top_score: float | None,
        runner_up_score: float | None,
        latency_ms: float,
        reason: str | None,
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO recognition_logs (
                    id, captured_at, decision, person_id, top_score,
                    runner_up_score, latency_ms, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    _now().isoformat(),
                    decision,
                    person_id,
                    top_score,
                    runner_up_score,
                    latency_ms,
                    reason,
                ),
            )

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS face_samples (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE RESTRICT,
                    image_path TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    pose TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
                    top_score REAL,
                    runner_up_score REAL,
                    latency_ms REAL NOT NULL,
                    reason TEXT
                );
                """
            )

    @staticmethod
    def _make_sample(
        *,
        person_id: str,
        image_path: str,
        embedding: np.ndarray,
        pose: str,
        quality: dict[str, float],
    ) -> FaceSample:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        return FaceSample(
            id=str(uuid4()),
            person_id=person_id,
            image_path=image_path,
            embedding=vector,
            pose=pose,
            quality=quality,
            created_at=_now(),
        )

    def _insert_sample(self, sample: FaceSample) -> None:
        self._connection.execute(
            """
            INSERT INTO face_samples (
                id, person_id, image_path, embedding_blob, embedding_dim,
                pose, quality_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.id,
                sample.person_id,
                sample.image_path,
                sample.embedding.tobytes(),
                sample.embedding.size,
                sample.pose,
                json.dumps(sample.quality, ensure_ascii=False),
                sample.created_at.isoformat(),
            ),
        )

    @staticmethod
    def _sample_from_row(row: sqlite3.Row) -> FaceSample:
        embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).copy()
        if embedding.size != row["embedding_dim"]:
            raise ValueError(f"stored embedding {row['id']} has an invalid dimension")
        return FaceSample(
            id=row["id"],
            person_id=row["person_id"],
            image_path=row["image_path"],
            embedding=embedding,
            pose=row["pose"],
            quality=json.loads(row["quality_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


def _now() -> datetime:
    return datetime.now(UTC)
