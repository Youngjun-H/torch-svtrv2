# svtrv2/example_infer.py
"""Example inference script"""
from pathlib import Path

from infer import SVTRv2Inference

# 설정
config_path = "configs/svtrv2_rctc.yml"
checkpoint_path = "/home/yjhwang/work/torch-svtrv2/last.ckpt"
# image_path = "path/to/image.jpg"

# Inference 초기화
inferencer = SVTRv2Inference(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device="cuda"
)

# # 단일 이미지 추론
# result = inferencer.predict(image_path)
# print(f"Predicted text: {result[0][0]}, Score: {result[0][1]:.4f}")

# 여러 이미지 배치 추론
image_list = [
    "/home/yjhwang/work/torch-svtrv2/01머3106_00021.jpg",
    "/home/yjhwang/work/torch-svtrv2/01보0550_00074.jpg",
    "/home/yjhwang/work/torch-svtrv2/05오0778_00003.jpg",
]
results = inferencer.predict_batch(image_list, batch_size=4)
for text, score in results:
    print(f"Text: {text}, Score: {score:.4f}")