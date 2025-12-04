import json
import random
import shutil
from pathlib import Path


def create_jsonl_from_lpdataset(dataset_dir, dataset_name, output_base_dir, seed=42):
    """
    LPDataset 디렉토리를 분석하여 train/val 데이터셋을 생성합니다.
    각 라벨(폴더)에서 랜덤으로 이미지 1장을 선택하여 validation set으로 분리합니다.
    단, 이미지가 5장 이하인 라벨의 경우 validation set을 만들지 않고 모든 이미지를 train으로 사용합니다.

    Args:
        dataset_dir: 원본 데이터셋 디렉토리 경로 (예: LPDataset_1120_raw)
        dataset_name: 데이터셋 이름 (예: "LPDataset_1120")
        output_base_dir: 출력할 기본 디렉토리 경로
        seed: 랜덤 시드 (기본값: 42)
    """
    random.seed(seed)

    dataset_path = Path(dataset_dir)
    output_base_path = Path(output_base_dir)

    # 데이터셋 디렉토리 구조 생성: dataset_name/train, dataset_name/val
    dataset_output_path = output_base_path / dataset_name
    train_path = dataset_output_path / "train"
    val_path = dataset_output_path / "val"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # train과 val 항목을 저장할 리스트
    train_entries = []
    val_entries = []

    train_copied_count = 0
    val_copied_count = 0
    label_count = 0

    # 먼저 유효한 라벨 디렉토리 목록 수집
    print("데이터셋 디렉토리 스캔 중...")
    valid_subdirs = []
    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue

        label_file = subdir / "label.txt"
        if not label_file.exists():
            continue

        with open(label_file, "r", encoding="utf-8") as f:
            label_text = f.read().strip()

        if not label_text:
            continue

        image_files = [
            f
            for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        if len(image_files) == 0:
            continue

        valid_subdirs.append((subdir, label_text, image_files))

    total_labels = len(valid_subdirs)
    print(f"총 {total_labels}개의 유효한 라벨 디렉토리를 찾았습니다.")
    print("데이터셋 처리 시작...\n")

    # 디렉토리 내의 모든 하위 디렉토리 순회
    for idx, (subdir, label_text, image_files) in enumerate(valid_subdirs, 1):
        # 진행 상황 출력 (10개마다 또는 마지막)
        if idx % 10 == 0 or idx == total_labels:
            progress = (idx / total_labels) * 100
            print(
                f"[{idx}/{total_labels}] 진행률: {progress:.1f}% - 현재 처리 중: {subdir.name}"
            )

        label_count += 1

        # 이미지 파일 정렬
        image_files = sorted(image_files)

        # 이미지가 5장 이하인 경우 validation set을 만들지 않고 모든 이미지를 train으로 사용
        if len(image_files) <= 5:
            # 모든 이미지를 train으로 처리
            for img_file in image_files:
                train_dest_path = train_path / img_file.name
                if not train_dest_path.exists():
                    shutil.copy2(img_file, train_dest_path)
                    train_copied_count += 1

                # JSONL 파일이 dataset_output_path에 있으므로, 상대 경로는 "train/파일명.jpg"
                train_relative_path = f"train/{img_file.name}"
                train_entry = {"filename": train_relative_path, "text": label_text}
                train_entries.append(train_entry)
        else:
            # 이미지가 6장 이상인 경우에만 validation set 생성
            # 각 라벨에서 랜덤으로 이미지 1장을 선택하여 validation set으로
            val_image = random.choice(image_files)
            train_images = [img for img in image_files if img != val_image]

            # Validation 이미지 복사 및 JSONL 항목 생성
            val_dest_path = val_path / val_image.name
            if not val_dest_path.exists():
                shutil.copy2(val_image, val_dest_path)
                val_copied_count += 1

            # JSONL 파일이 dataset_output_path에 있으므로, 상대 경로는 "val/파일명.jpg"
            val_relative_path = f"val/{val_image.name}"
            val_entry = {"filename": val_relative_path, "text": label_text}
            val_entries.append(val_entry)

            # Train 이미지들 복사 및 JSONL 항목 생성
            for img_file in train_images:
                train_dest_path = train_path / img_file.name
                if not train_dest_path.exists():
                    shutil.copy2(img_file, train_dest_path)
                    train_copied_count += 1

                # JSONL 파일이 dataset_output_path에 있으므로, 상대 경로는 "train/파일명.jpg"
                train_relative_path = f"train/{img_file.name}"
                train_entry = {"filename": train_relative_path, "text": label_text}
                train_entries.append(train_entry)

    # JSONL 파일로 저장
    print("\nJSONL 파일 저장 중...")
    train_jsonl_path = dataset_output_path / "train.jsonl"
    val_jsonl_path = dataset_output_path / "val.jsonl"

    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_jsonl_path, "w", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("=" * 80)
    print("처리 완료!")
    print("=" * 80)
    print(f"총 {label_count}개의 라벨이 처리되었습니다.")
    print(
        f"Train: {len(train_entries)}개 항목, {train_copied_count}개 이미지 파일이 {train_path}로 복사되었습니다."
    )
    print(
        f"Val: {len(val_entries)}개 항목, {val_copied_count}개 이미지 파일이 {val_path}로 복사되었습니다."
    )
    print(f"Train JSONL: {train_jsonl_path}")
    print(f"Val JSONL: {val_jsonl_path}")

    return train_entries, val_entries


if __name__ == "__main__":
    # 원본 데이터셋 디렉토리 경로
    dataset_dir = "/purestorage/AILAB/AI_2/datasets/LPR/raw/LPDataset_v1.0.0"

    # 데이터셋 이름 (원본 디렉토리명에서 _raw를 제거하거나 직접 지정)
    dataset_name = "v1.0.0"

    # 출력할 기본 디렉토리 경로
    output_base_dir = "/purestorage/AILAB/AI_2/datasets/LPR/processed"

    create_jsonl_from_lpdataset(dataset_dir, dataset_name, output_base_dir)
