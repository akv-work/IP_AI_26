#!/usr/bin/env python3
"""ЛР3 Яшков П.А., вариант 4 — воспроизводимый запуск YOLO11s.

Пример:
 python main.py --mode eval --data /path/to/data.yaml --weights /path/to/best.pt --output ./results
 python main.py --mode predict --images ./images --weights /path/to/best.pt --output ./results
 python main.py --mode train --data /path/to/data.yaml --weights yolo11s.pt --output ./results
"""
import argparse, csv, json
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',required=True,choices=['train','eval','predict'])
    ap.add_argument('--data',type=Path,help='YAML конфигурация датасета YOLO')
    ap.add_argument('--images',type=Path,help='Папка с фотографиями JPG/PNG')
    ap.add_argument('--weights',default='yolo11s.pt',help='Путь к файлу модели')
    ap.add_argument('--output',type=Path,default=Path('results'))
    ap.add_argument('--device',default=None,help='Например 0, cpu')
    args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else 'cpu')
    model=YOLO(str(args.weights))
    if args.mode in ('train','eval') and (args.data is None or not args.data.is_file()):
        ap.error('Для train/eval нужен --data /path/to/data.yaml')
    if args.mode == 'train':
        if device=='cpu':
            print('Внимание: обучение YOLO11s на CPU может быть очень медленным')
        model.train(data=str(args.data),epochs=30,imgsz=640,batch=16,device=device,
            workers=0,optimizer='auto',patience=0,save=True,save_period=5,plots=True,
            project=str(args.output/'training'),name='yolo11s_cats_v3',exist_ok=True,seed=42,amp=True)
    elif args.mode=='eval':
        m=model.val(data=str(args.data),split='test',imgsz=640,batch=16,device=device,
            workers=0,plots=True,project=str(args.output),name='test_evaluation',exist_ok=True)
        metrics={'precision':float(m.box.mp),'recall':float(m.box.mr),
            'mAP50':float(m.box.map50),'mAP75':float(m.box.map75),'mAP50-95':float(m.box.map)}
        (args.output/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(metrics)
    else:
        if args.images is None or not args.images.is_dir():
            ap.error('Для predict нужна папка --images')
        images=sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if not images:ap.error('В папке --images нет изображений')
        rs=model.predict(source=[str(p) for p in images],imgsz=640,conf=0.25,device=device,
            save=True,save_txt=True,save_conf=True,project=str(args.output),
            name='internet_predictions',exist_ok=True)
        rows=[]
        for r in rs:
            scores=[float(v) for v in r.boxes.conf]
            rows.append([Path(r.path).name,len(scores),round(max(scores)*100,2) if scores else 0])
        with (args.output/'internet_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['Изображение','Обнаружено котов','Макс. уверенность, %']);w.writerows(rows)
        print('Обработано изображений:',len(rows))
if __name__=='__main__':main()
