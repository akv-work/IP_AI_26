import os
import glob
import random
import matplotlib.pyplot as plt
from IPython.display import Image, display
from google.colab import drive
from roboflow import Roboflow
from ultralytics import YOLO

drive.mount('/content/drive')

save_dir = "/content/drive/MyDrive/YOLO_WaterMeter"
os.makedirs(save_dir, exist_ok=True)

rf = Roboflow(api_key="SUPER-SECRET-API-KEY")
project = rf.workspace("koer3741-gmail-com").project("watermeteramrv2")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolo12m.pt")

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    workers=2,
    cache=False,
    device=0,
    project=save_dir,
    name="experiment_v1",
    exist_ok=True
)

exp_path = f"{save_dir}/experiment_v1"

results_png = os.path.join(exp_path, "results.png")
matrix_png = os.path.join(exp_path, "confusion_matrix.png")

if os.path.exists(results_png):
    display(Image(filename=results_png))

if os.path.exists(matrix_png):
    display(Image(filename=matrix_png))

best_model_path = os.path.join(exp_path, "weights", "best.pt")
trained_model = YOLO(best_model_path)

test_images = glob.glob(f"{dataset.location}/test/images/*.jpg") + \
              glob.glob(f"{dataset.location}/test/images/*.png")

if test_images:
    random_image_path = random.choice(test_images)
    
    pred_results = trained_model.predict(source=random_image_path, conf=0.25)
    
    for res in pred_results:
        res_bgr = res.plot()
        res_rgb = res_bgr[:, :, ::-1]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(res_rgb)
        plt.axis('off')
        plt.show()