# svtrv2/example_infer.py
"""Example inference script"""
from infer import SVTRv2Inference

# 설정
config_path = "/home/yjhwang/work/torch-svtrv2/svtrv2/configs/svtrv2_rctc.yml"
checkpoint_path = "/home/yjhwang/work/torch-svtrv2/svtrv2/output/svtrv2_rctc/1127_v3/checkpoint-epochepoch=77-val_loss=0.442.ckpt"
# checkpoint_path = "/home/yjhwang/work/torch-svtrv2/checkpoint-epoch=44-val_loss=0.000.ckpt"
# image_path = "path/to/image.jpg"

# Inference 초기화
inferencer = SVTRv2Inference(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device="cuda"
)


# 여러 이미지 배치 추론
image_list = [
    "/home/yjhwang/work/torch-svtrv2/01머3106_00021.jpg",
    "/home/yjhwang/work/torch-svtrv2/01보0550_00074.jpg",
    "/home/yjhwang/work/torch-svtrv2/05오0778_00003.jpg",
    "/home/yjhwang/work/torch-svtrv2/253거5032.jpg",
]
results = inferencer.predict_batch(image_list, batch_size=4)
for text, score in results:
    print(f"Text: {text}, Score: {score:.4f}, length: {len(text)}")