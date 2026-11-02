from google.colab import files
import cv2
import matplotlib.pyplot as plt
 
up = files.upload()
for name in up:
    r = best.predict(name, conf=0.25, verbose=False)[0]
    img = r.plot()
    cv2.imwrite("pred_" + name, img)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(name)
    plt.show()
