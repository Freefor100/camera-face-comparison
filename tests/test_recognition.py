from __future__ import annotations

import hashlib

import numpy as np

from camera_face_comparison.config import load_settings
from camera_face_comparison.face_engine import FaceInputError, FaceObservation
from camera_face_comparison.face_library import InMemoryFaceLibrary
from camera_face_comparison.image_input import ImageInput
from camera_face_comparison.open_set_policy import ScoreThresholdPolicy
from camera_face_comparison.recognition import RecognitionService
from camera_face_comparison.repository import FaceRepository, SampleInput


def _sample_input(
    embedding: np.ndarray,
    *,
    image_path: str,
) -> SampleInput:
    """构造人员原型测试使用的样本输入。"""

    return SampleInput(
        image_path=image_path,
        embedding=embedding,
        quality_metrics={},
        source_type="file",
    )


def test_mean_prototype_uses_all_reference_embeddings_before_scoring(tmp_path) -> None:
    """人员分数必须来自归一化平均原型，而不是最高样本或 Top-K 分数。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice",
        display_name="Alice",
        samples=[
            _sample_input(np.array([1.0, 0.0], dtype=np.float32), image_path="faces/alice/1.jpg"),
            _sample_input(
                np.array([0.10, 0.995], dtype=np.float32), image_path="faces/alice/2.jpg"
            ),
        ],
    )
    repository.create_person_with_samples(
        person_id="bob",
        display_name="Bob",
        samples=[
            _sample_input(np.array([0.91, 0.414], dtype=np.float32), image_path="faces/bob/1.jpg"),
            _sample_input(np.array([0.90, 0.436], dtype=np.float32), image_path="faces/bob/2.jpg"),
        ],
    )
    decision = InMemoryFaceLibrary.from_repository(repository).snapshot().search(
        np.array([1.0, 0.0], dtype=np.float32), ScoreThresholdPolicy(minimum_score=0.80)
    )
    repository.close()

    assert decision.accepted_person_id == "bob"
    assert decision.rule == "score_threshold"
    assert decision.acceptance_score == decision.top_score


def test_frozen_score_rule_does_not_reject_a_close_second_candidate(tmp_path) -> None:
    """部署规则只检查最高分，候选分差只用于解释结果。"""

    repository = FaceRepository(tmp_path / "face_library.sqlite")
    repository.create_person_with_samples(
        person_id="alice",
        display_name="Alice",
        samples=[_sample_input(np.array([0.90, 0.43589], dtype=np.float32), image_path="faces/alice/1.jpg")],
    )
    repository.create_person_with_samples(
        person_id="bob",
        display_name="Bob",
        samples=[_sample_input(np.array([0.89, 0.45596], dtype=np.float32), image_path="faces/bob/1.jpg")],
    )
    decision = InMemoryFaceLibrary.from_repository(repository).snapshot().search(
        np.array([1.0, 0.0], dtype=np.float32), ScoreThresholdPolicy(minimum_score=0.80)
    )
    repository.close()

    assert decision.accepted_person_id == "alice"
    assert decision.score_gap is not None and decision.score_gap < 0.02


def test_recognition_service_returns_calibrated_fields_and_quality_warnings(tmp_path) -> None:
    """低质量数值不阻断单脸，结果需同时返回策略字段和调整提示。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    timings: dict[str, list[float]] = {}
    alice = _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    _create_person(
        repository,
        settings,
        "Bob",
        [np.array([0.2, 0.98], dtype=np.float32)],
    )
    library = InMemoryFaceLibrary.from_repository(repository)

    class ProbeEngine:
        """返回低质量指标但 embedding 有效的 Alice 特征。"""

        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            """构造可继续识别的单脸观察。"""

            return FaceObservation(
                bbox=(0.0, 0.0, 32.0, 32.0),
                detection_score=0.30,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                blur_variance=1.0,
                landmarks=None,
            )

    result = RecognitionService(
        repository,
        settings,
        ProbeEngine(),
        library.snapshot(),
        timing_sink=lambda stage, value: timings.setdefault(stage, []).append(value),
    ).compare(np.full((80, 80, 3), 5, dtype=np.uint8))
    repository.close()

    assert result.status == "matched"
    assert result.person_id == alice.id
    assert result.display_name == "Alice"
    assert result.acceptance_rule == "score_threshold"
    assert result.acceptance_score == result.top_score
    assert result.score_gap is not None
    assert "move_closer" in result.quality_warnings
    assert "hold_still" in result.quality_warnings
    assert "increase_lighting" in result.quality_warnings
    assert set(timings) == {
        "face_inference_ms",
        "quality_measurement_ms",
        "candidate_search_ms",
        "decision_ms",
        "log_write_ms",
    }
    assert all(values and values[0] >= 0.0 for values in timings.values())


def test_recognition_service_uses_snapshot_without_reading_samples(tmp_path, monkeypatch) -> None:
    """识别查询应只使用启动时传入的快照，不在热路径读取 SQLite 样本。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    library = InMemoryFaceLibrary.from_repository(repository)

    def fail(*args, **kwargs):
        raise AssertionError("recognition query must not read the repository gallery")

    monkeypatch.setattr(repository, "list_samples", fail)
    monkeypatch.setattr(repository, "list_people", fail)

    class ProbeEngine:
        """返回 Alice 特征的最小测试引擎。"""

        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            """返回一张有效单脸观察。"""

            return FaceObservation(
                bbox=(0.0, 0.0, 64.0, 64.0),
                detection_score=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                blur_variance=100.0,
                landmarks=None,
            )

    result = RecognitionService(
        repository,
        settings,
        ProbeEngine(),
        library.snapshot(),
    ).compare(np.zeros((80, 80, 3), dtype=np.uint8))
    repository.close()

    assert result.status == "matched"
    assert result.person_id == "alice"


def test_recognition_service_selects_sharpest_valid_frame_and_skips_invalid_frames(tmp_path) -> None:
    """多帧识别应跳过无效帧，并用清晰度最高的有效帧完成一次判定。"""

    settings = load_settings(tmp_path)
    repository = FaceRepository(settings.database_path)
    _create_person(
        repository,
        settings,
        "Alice",
        [np.array([1.0, 0.0], dtype=np.float32)],
    )
    library = InMemoryFaceLibrary.from_repository(repository)

    class BurstEngine:
        """根据帧像素返回不同清晰度的 Alice 观察。"""

        def extract_single_face(self, frame: np.ndarray) -> FaceObservation:
            """第二帧模拟检测失败，其余帧返回有效特征。"""

            if int(frame[0, 0, 0]) == 1:
                raise FaceInputError("multiple_faces")
            return FaceObservation(
                bbox=(0.0, 0.0, 64.0, 64.0),
                detection_score=0.95,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                blur_variance=float(frame[0, 0, 0] * 10 + 10),
                landmarks=None,
            )

    inputs = [
        ImageInput.from_camera(np.full((80, 80, 3), value, dtype=np.uint8))
        for value in range(5)
    ]
    result = RecognitionService(
        repository,
        settings,
        BurstEngine(),
        library.snapshot(),
    ).compare_inputs(inputs)
    repository.close()

    assert result.status == "matched"
    assert result.person_id == "alice"
    assert result.frame_count == 5
    assert result.valid_frame_count == 4
    assert result.selected_frame_index == 4
    assert result.selected_method == "sharpest_frame"


def _create_person(
    repository: FaceRepository,
    settings,
    name: str,
    embeddings: list[np.ndarray],
):
    """为识别服务测试创建一个带指定特征和真实文件的人员。"""

    person_id = name.lower()
    samples = []
    for index, embedding in enumerate(embeddings, start=1):
        relative_path = f"faces/{person_id}/sample_{index:03d}.jpg"
        image_path = settings.data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"{name}-{index}".encode())
        samples.append(
            SampleInput(
                image_path=relative_path,
                embedding=embedding,
                quality_metrics={
                    "detection_score": 0.95,
                    "face_size_px": 160.0,
                    "blur_variance": 150.0,
                    "brightness": 120.0,
                    "contrast": 20.0,
                },
                source_type="file",
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
        )
    return repository.create_person_with_samples(
        person_id=person_id,
        display_name=name,
        samples=samples,
    )
