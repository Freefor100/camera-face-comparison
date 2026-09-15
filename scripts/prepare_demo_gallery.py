from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_face_comparison.config import load_settings
from camera_face_comparison.enrollment import EnrollmentService
from camera_face_comparison.experiment_artifacts import write_json_atomic
from camera_face_comparison.face_engine import FaceEngine
from camera_face_comparison.image_input import ImageInput, measure_quality, quality_warnings
from camera_face_comparison.repository import FaceRepository

DEMO_PEOPLE: dict[str, tuple[str, ...]] = {
    "Angela Merkel": ("angela_merkel/angela_merkel_2011.jpg",),
    "Barack Obama": (
        "barack_obama/barack_obama_2009.jpg",
        "barack_obama/barack_obama_2012.png",
        "barack_obama/barack_obama_2012_original.jpg",
        "barack_obama/barack_obama_2016.jpg",
    ),
    "Donald Trump": ("donald_trump/donald_trump_2017.jpg",),
    "景甜": (
        "jingtian/jingtian_01.jpeg",
        "jingtian/jingtian_02.jpeg",
        "jingtian/jingtian_03.jpeg",
        "jingtian/jingtian_04.jpeg",
    ),
    "孙宇晨": (
        "sun_yuchen/sun_yuchen_01.jpeg",
        "sun_yuchen/sun_yuchen_02.jpeg",
        "sun_yuchen/sun_yuchen_03.jpeg",
        "sun_yuchen/sun_yuchen_04.jpeg",
        "sun_yuchen/sun_yuchen_05.jpeg",
    ),
    "张继科": (
        "zhangjike/zhangjike_01.jpeg",
        "zhangjike/zhangjike_02.jpg",
        "zhangjike/zhangjike_03.jpg",
    ),
}


def parse_args() -> argparse.Namespace:
    """读取最终演示标准库建立参数。"""
    parser = argparse.ArgumentParser(description="使用候选图片建立最终演示标准库")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--candidates-dir", type=Path, default=Path("data/demo-candidates")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/experiments/phase6/demo-gallery-manifest.json"),
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="允许在已有标准库中追加本批演示人员；默认要求数据库为空",
    )
    return parser.parse_args()


def main() -> int:
    """预检候选图片并通过正式录入服务建立演示标准库。"""
    args = parse_args()
    data_dir = args.data_dir.resolve()
    candidates_dir = args.candidates_dir.resolve()
    settings = load_settings(data_dir)
    repository = FaceRepository(settings.database_path)
    try:
        existing_people = repository.list_people()
        if existing_people and not args.allow_existing:
            raise RuntimeError(
                f"demo gallery database is not empty ({len(existing_people)} people); "
                "use a new data directory or pass --allow-existing"
            )

        engine = FaceEngine.from_local_model(settings)
        inputs_by_person: dict[str, list[ImageInput]] = {}
        preflight: dict[str, list[dict[str, object]]] = {}
        for display_name, relative_paths in DEMO_PEOPLE.items():
            person_inputs: list[ImageInput] = []
            person_rows: list[dict[str, object]] = []
            for relative_path in relative_paths:
                path = candidates_dir / relative_path
                image_input = ImageInput.from_file(path, source_type="file")
                observation = engine.extract_single_face(image_input.frame)
                metrics = measure_quality(image_input.frame, observation)
                person_inputs.append(image_input)
                person_rows.append(
                    {
                        "relative_path": relative_path,
                        "width": int(image_input.frame.shape[1]),
                        "height": int(image_input.frame.shape[0]),
                        "quality_metrics": metrics,
                        "quality_warnings": list(
                            quality_warnings(metrics, settings.quality_warnings)
                        ),
                    }
                )
            inputs_by_person[display_name] = person_inputs
            preflight[display_name] = person_rows

        service = EnrollmentService(
            repository=repository,
            settings=settings,
            face_engine=engine,
            image_saver=_save_bgr_image,
        )
        enrolled: list[dict[str, object]] = []
        for display_name, inputs in inputs_by_person.items():
            person = service.create_from_inputs(display_name, inputs)
            enrolled.append(
                {
                    "person_id": person.id,
                    "display_name": person.display_name,
                    "sample_count": len(inputs),
                    "source_paths": [row["relative_path"] for row in preflight[display_name]],
                }
            )
    finally:
        repository.close()

    manifest = {
        "artifact": "phase6-demo-gallery-v1",
        "model_backend": {
            "name": engine.backend.name,
            "providers": list(engine.backend.providers),
        },
        "data_dir": str(data_dir),
        "candidates_dir": str(candidates_dir),
        "people": enrolled,
        "preflight": preflight,
        "notes": [
            "演示标准库在阈值、聚合方法和界面冻结后建立。",
            "每个人员至少一张有效图片即可参与识别；质量异常只保留提示，不阻断有效单脸。",
            "预检全部通过后才开始写入；六个人员之间不是一个跨人员事务。",
        ],
    }
    write_json_atomic(args.manifest.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _save_bgr_image(path: Path, frame) -> None:
    """使用 OpenCV 保存录入服务准备写入标准库的 BGR 图片。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is not installed; install the project dependencies first") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not save image to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
