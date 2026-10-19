from ultralytics import YOLO

def main():
    # Load pretrained model YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Jalankan training dengan augmentasi data
    model.train(
        data="C:/Games/smart_shelf/dataset/data.yaml",
        epochs=150,
        imgsz=640,
        batch=4,
        
        # Heavy Data Augmentation
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=10.0,
        perspective=0.0005,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.2,
        erasing=0.4,
        
        project="C:/Games/smart_shelf/runs",
        name="few_shot_experiment"
    )

if __name__ == "__main__":
    main()