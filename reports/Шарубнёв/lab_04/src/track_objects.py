import cv2
from ultralytics import YOLO

def main():
    model_path = r'E:\PythonProjects\OVIS3\runs\detect\yolo12s_vehicles-2\weights\best.pt'
    model = YOLO(model_path)

    video_source = 'video1.mp4'

    print("Запуск ByteTrack (стандартный)...")
    results_bytetrack = model.track(
        source=video_source,
        tracker="bytetrack.yaml",
        save=True,
        project="runs/track",
        name="bytetrack_default",
        stream=True
    )
    for _ in results_bytetrack:
        pass

    print("Запуск ByteTrack (кастомный)...")
    results_custom = model.track(
        source=video_source,
        tracker="custom_bytetrack.yaml",
        save=True,
        project="runs/track",
        name="bytetrack_custom",
        stream=True
    )
    for _ in results_custom:
        pass

    print("Запуск BoT-Sort (стандартный)...")
    results_botsort = model.track(
        source=video_source,
        tracker="botsort.yaml",
        save=True,
        project="runs/track",
        name="botsort_default",
        stream=True
    )
    for _ in results_botsort:
        pass

if __name__ == '__main__':
    main()