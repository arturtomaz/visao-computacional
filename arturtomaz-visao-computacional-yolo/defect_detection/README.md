# YOLO defect detection

Esta pasta implementa a etapa de deteccao de falhas em etiquetas metalicas:

- `ranhura`
- `amassado`
- `mancha`

A ideia e usar a etapa ja existente do projeto para localizar/retificar a etiqueta e, em seguida, usar YOLO para detectar as falhas na area da etiqueta.

## 1. Preparar imagens para anotacao

Use o script abaixo para gerar recortes retificados das etiquetas. Esses recortes sao melhores para anotar porque removem fundo, perspectiva e escala variavel.

```bash
python -m defect_detection.prepare_label_crops tendence capotas indialar_moveis --output datasets/falhas/images_to_annotate --debug
```

Depois anote as imagens em uma ferramenta como LabelImg, CVAT ou Roboflow usando as classes:

```text
0 ranhura
1 amassado
2 mancha
```

## 2. Organizar dataset YOLO

Coloque as imagens e labels assim:

```text
datasets/falhas/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Cada imagem deve ter um `.txt` correspondente em `labels/...`:

```text
classe x_center y_center width height
```

As coordenadas precisam estar normalizadas de 0 a 1.

## 3. Treinar

```bash
python -m defect_detection.train --data datasets/falhas.yaml --model yolo26n.pt --epochs 100 --imgsz 960 --batch 8
```

Se o computador for fraco ou sem GPU:

```bash
python -m defect_detection.train --data datasets/falhas.yaml --model yolo26n.pt --epochs 50 --imgsz 640 --batch 4 --device cpu
```

O melhor peso fica em:

```text
runs/defects/train/weights/best.pt
```

## 4. Rebalancear dataset exportado do Roboflow

Se o Roboflow gerou um dataset muito desbalanceado, ou se a validacao/teste ficaram pequenos demais, use:

```bash
python -m defect_detection.rebalance_yolo_dataset --source "datasets/Object Detection.v1-dataset-yolo.yolov8" --output datasets/falhas_rebalanced --train 0.70 --val 0.20 --test 0.10 --max-background-ratio 0.30
```

O script junta `train`, `valid` e `test`, redistribui as imagens por classe, limita imagens sem defeito e gera um `data.yaml` novo. Ele tambem converte labels de poligono/segmentacao para bounding boxes, evitando dataset misto detect/segment.

Treine com:

```bash
python -m defect_detection.train --data datasets/falhas_rebalanced/data.yaml --model yolov8s.pt --epochs 100 --imgsz 960 --batch 16
```

## 5. Predizer

Em imagens ja recortadas/retificadas:

```bash
python -m defect_detection.predict --weights runs/defects/train/weights/best.pt --source datasets/falhas/images/test
```

Em fotos originais, usando primeiro a retificacao do projeto:

```bash
python -m defect_detection.predict --weights runs/defects/train/weights/best.pt --source tendence --rectify --debug
```

As saidas ficam em:

```text
runs/defects/predict/
  predictions/
  detections.json
```

## O que defender na apresentacao

O YOLO aprende as falhas a partir de exemplos anotados. A etapa atual do projeto ja encontra a etiqueta e corrige perspectiva; a minha etapa usa esse resultado como entrada para detectar defeitos locais. Isso reduz ruido de fundo e deixa a rede focar no que importa: ranhura, amassado e mancha.
