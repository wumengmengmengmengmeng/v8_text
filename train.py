# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import time

from ultralytics import YOLO

if __name__ == "__main__":
    start_time = time.time()

    model = YOLO("./yolov8n.pt")  # 可以改成你的本地预训练权重路径

    results = model.train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        workers=0,
        batch=-1,
        device="",
        resume=False,
        project="runs/train",
        name="exp2",  # 注意和你要读取的 best.pt 对应
        single_cls=False,
        cache=True,
    )
