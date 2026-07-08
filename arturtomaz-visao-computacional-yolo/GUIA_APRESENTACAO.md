# Guia de apresentacao do projeto

Este repositorio implementa um calculador de proporcao e verificacao de cor para etiquetas metalicas em imagens. O arquivo central e `ratio_calculator.py`; o notebook `Untitled.ipynb` e praticamente so configuracao de ambiente/GPU e nao participa da solucao principal.

Observacao importante: o arquivo anexado em `C:\Users\gabri\Downloads\ratio_calculator.py` esta vazio, com 0 bytes. A versao analisavel e a do repositorio clonado.

## Ideia principal

O problema e medir a razao largura:altura de uma etiqueta metalica e comparar a cor impressa com uma referencia, mesmo quando a foto tem perspectiva, sombra, iluminacao diferente e cantos arredondados.

A solucao evita usar diretamente os vertices detectados por contorno, porque etiquetas com cantos arredondados confundem algoritmos como `approxPolyDP`: eles podem colocar vertices nas curvas. O script procura os quatro lados retos, ajusta uma reta em cada lado e calcula os cantos pela intersecao dessas retas.

## Como rodar

Instale as dependencias:

```bash
pip install -r requirements
```

Execute passando a pasta de uma etiqueta:

```bash
python ratio_calculator.py tendence
python ratio_calculator.py capotas
python ratio_calculator.py indialar_moveis
```

O script cria/atualiza a pasta `debug` dentro da pasta analisada e grava imagens auxiliares e `log.txt`.

## Estrutura do repositorio

O projeto tem um unico script Python relevante e varias pastas de imagens:

| Pasta | Template real | Fake template | Fotos processaveis |
| --- | ---: | ---: | ---: |
| `busscar` | 1 | 0 | 2 |
| `capotas` | 0 | 1 | 7 |
| `gem` | 0 | 0 | 1 |
| `houseflex` | 0 | 0 | 1 |
| `hydra_alphard` | 0 | 1 | 6 |
| `indialar_moveis` | 0 | 1 | 7 |
| `la_poltrona` | 0 | 0 | 1 |
| `santa_catarina` | 0 | 1 | 2 |
| `so_bercos` | 1 | 0 | 2 |
| `tendence` | 1 | 0 | 9 |

Pastas sem `template.*` ou `fake_template.*` nao conseguem usar todo o fluxo de comparacao com referencia.

## Fluxo completo

1. Entrada

O usuario passa uma pasta. O script procura:

- `template.*`: imagem limpa da etiqueta, geralmente em fundo branco.
- `fake_template.*`: uma imagem usada como referencia quando nao existe template limpo.
- Fotos reais: arquivos `.jpg`, `.jpeg` ou `.png` que nao tenham `template` no nome.

2. Template

Funcao: `calculate_template_ratio`.

O script carrega o template, converte para cinza, aplica threshold para separar objeto e fundo branco, remove bordas artificiais e pega o maior contorno.

Depois chama `find_true_corners` para obter os quatro cantos reais por intersecao de retas. Com os cantos, calcula:

- largura media: media entre topo e base.
- altura media: media entre esquerda e direita.
- ratio: largura media / altura media.

Se a deteccao de linhas falhar, usa o ratio bruto de pixels da imagem.

3. Cantos verdadeiros

Funcao: `find_true_corners`.

Etapas:

- Usa `cv2.minAreaRect` para obter um retangulo orientado aproximado.
- Ordena os pontos em TL, TR, BR, BL.
- Para cada lado, pega pontos do contorno que ficam no meio do lado, evitando os cantos arredondados.
- Usa `cv2.fitLine` para ajustar uma reta em cada lado.
- Calcula as quatro intersecoes:
  - topo com esquerda = TL
  - topo com direita = TR
  - direita com base = BR
  - base com esquerda = BL
- Faz uma checagem de area: a area dos cantos calculados precisa ser parecida com a area do contorno.

Esse e o ponto tecnico mais importante para explicar: em vez de aceitar vertices falsos nos cantos arredondados, ele reconstrui os cantos a partir dos lados retos.

4. Deteccao da etiqueta na foto

Funcao: `detect_label_in_photo`.

O script transforma a foto em escala de cinza, melhora contraste com CLAHE e aplica blur. Depois tenta achar o melhor contorno retangular com `_find_best_contour`.

`_find_best_contour` testa varias estrategias:

- Canny com diferentes limiares.
- Threshold adaptativo.
- Threshold de Otsu.
- Fechamentos morfologicos.

Cada contorno recebe uma pontuacao baseada em:

- area minima e maxima.
- retangularidade.
- compatibilidade com o ratio esperado do template.
- penalidade se o contorno ocupa area demais da imagem.
- contraste real da borda, para evitar pegar sombra de parede como se fosse etiqueta.

Se a primeira tentativa falha, o script tenta normalizar sombras dividindo a imagem por uma versao borrada dela mesma, depois repete a busca.

5. Calculo da proporcao nas fotos

Funcao: `compute_aspect_ratio`.

O script calcula dois ratios:

- `simple_ratio`: media das larguras dividida pela media das alturas.
- `zhang_ratio`: tentativa de correcao 3D usando pontos de fuga.

O metodo Zhang usa as intersecoes das linhas paralelas da etiqueta para estimar pontos de fuga horizontal e vertical. A partir deles, estima uma distancia focal e reconstrui os quatro cantos em uma geometria 3D simplificada.

Se o valor 3D parece valido, o script usa `Zhang 3D`. Se nao, volta para o metodo simples.

6. Comparacao de cor

Funcoes principais:

- `prepare_template_color_reference`
- `_build_template_masks`
- `_normalize_photo_color`
- `detect_color_change`

O template e retificado para uma vista frontal. Depois o script cria mascaras:

- `label_mask`: toda a etiqueta.
- `base_mask`: base metalica.
- `print_mask`: regioes impressas, como texto e logo.

Na foto, a etiqueta tambem e retificada com `warpPerspective`. A cor da foto e normalizada em LAB usando a base metalica como referencia. A ideia e compensar diferencas de iluminacao/cast de cor antes de julgar a impressao.

Depois o script compara os canais cromaticos `a` e `b` do LAB:

- `mean_base_delta`: diferenca media na base metalica.
- `mean_print_delta`: diferenca media nas regioes impressas.
- `mean_color_print_delta`: diferenca media em regioes impressas coloridas/saturadas.
- `p90_color_print_delta`: percentil 90 da diferenca nessas regioes.
- `local_color_component_score`: maior diferenca media em um componente conectado relevante.

A cor e marcada como alterada se qualquer regra passar:

- `effective_delta > 8.0`
- `mean_print_delta > 10.0` e `max_print_delta > 18.0`
- `mean_color_print_delta > 4.5` e `p90_color_print_delta > 8.0`
- componente local com media > 5.0, p95 > 7.5 e area >= 150 px

`effective_delta` e calculado como:

```text
max(0, mean_print_delta - 0.6 * mean_base_delta)
```

Ou seja: se a base inteira mudou por iluminacao, isso e parcialmente descontado. O que sobra na impressao e mais suspeito.

## Imagens de debug

Os arquivos `debug_*.jpg` ajudam muito na apresentacao.

Exemplos:

- `debug_template.jpg`: mostra template com cantos/linhas.
- `debug_template_color_reference.jpg`: verde = base metalica; vermelho = texto/logo.
- `debug_<foto>.jpg`: contorno e cantos detectados na foto.
- `debug_<foto>_simples.jpg`: ratio simples.
- `debug_<foto>_zhang.jpg`: ratio com Zhang 3D, quando disponivel.
- `debug_<foto>_findContours.jpg`: contorno bruto encontrado.
- `debug_<foto>_contourArea.jpg`: area preenchida do contorno.
- `debug_<foto>_color_normalized.jpg`: foto retificada e normalizada em cor.
- `debug_<foto>_color_heatmap.jpg`: mapa de diferenca de cor com status `OK` ou `ALTERADA`.

Exemplos observados:

- `tendence/debug/debug_tendence1_color_heatmap.jpg`: status `OK`.
- `hydra_alphard/debug/debug_hydra_alphard1_color_heatmap.jpg`: status `OK`.
- `capotas/debug/debug_capotas1_color_heatmap.jpg`: status `ALTERADA`.
- `indialar_moveis/debug/debug_afaef_color_heatmap.jpg`: status `ALTERADA`.

## Roteiro sugerido para apresentar

1. Comece pelo problema: medir etiquetas metalicas em fotos reais e verificar se a impressao/cor bate com a referencia.
2. Mostre por que nao basta medir pixels: a foto pode ter perspectiva, sombra, iluminacao e cantos arredondados.
3. Explique a escolha principal: detectar os quatro lados retos e calcular os cantos pela intersecao das retas.
4. Mostre uma imagem `debug_<foto>.jpg` com TL/TR/BR/BL.
5. Explique os dois ratios:
   - simples: media dos lados na imagem.
   - Zhang 3D: tenta corrigir perspectiva por pontos de fuga.
6. Mostre o mapa de cor:
   - normaliza iluminacao usando a base metalica.
   - compara a impressao no espaco LAB.
   - classifica como `OK` ou `ALTERADA`.
7. Feche com limitacoes e proximos passos.

## Limitacoes

- Depende de thresholds manuais; pode precisar ajuste para novos tipos de etiqueta.
- Assume que a etiqueta e aproximadamente retangular.
- Templates reais esperam fundo claro/branco.
- Fotos com borda pouco contrastada ou etiqueta muito pequena podem falhar.
- O metodo Zhang 3D so e usado quando os pontos de fuga permitem uma estimativa valida.
- A deteccao de cor e heuristica; nao e calibracao colorimetrica profissional.
- Nao ha README nem testes automatizados no repositorio.

## Perguntas provaveis

Por que usar LAB para cor?

LAB separa melhor luminancia de cromaticidade. O script usa os canais `a` e `b` para comparar cor, reduzindo o impacto de brilho.

Por que normalizar pela base metalica?

Porque a base metalica sofre a mesma iluminacao da impressao. Se a base mudou por luz/sombra, o script corrige a foto antes de comparar a cor impressa.

Por que nao usar `approxPolyDP`?

Porque os cantos arredondados geram vertices errados na curva. Ajustar retas aos lados retos e intersectar essas retas e mais robusto.

O que e `fake_template`?

E uma referencia alternativa quando nao existe um template limpo. O script detecta a etiqueta nessa imagem e a usa como base para ratio/cor.

O que significa `best_method`?

Se a reconstrucao 3D for valida, usa `Zhang 3D`; caso contrario, usa `simples`.

Qual e o ponto forte do projeto?

A combinacao de geometria classica de visao computacional com debugs visuais claros: e possivel inspecionar onde a deteccao acertou ou falhou.

## Nova etapa: deteccao de falhas com YOLO

A nova parte do projeto adiciona um detector supervisionado para falhas locais na etiqueta:

- `ranhura`
- `amassado`
- `mancha`

O YOLO nao substitui a etapa anterior; ele entra depois dela. A etapa antiga localiza a etiqueta e corrige perspectiva. A etapa nova usa a etiqueta ja recortada/retificada como entrada para detectar defeitos com bounding boxes.

Fluxo recomendado:

```text
foto original
  -> detectar etiqueta
  -> retificar etiqueta com perspectiva corrigida
  -> YOLO detecta ranhura/amassado/mancha
  -> salvar imagem com caixas + JSON com resultados
```

Arquivos adicionados:

- `defect_detection/prepare_label_crops.py`: gera imagens retificadas para anotacao manual.
- `defect_detection/train.py`: treina o YOLO.
- `defect_detection/predict.py`: roda inferencia em imagens recortadas ou em fotos originais com `--rectify`.
- `datasets/falhas.yaml`: configuracao das classes e pastas do dataset.

Comandos principais:

```bash
python -m defect_detection.prepare_label_crops tendence capotas indialar_moveis --output datasets/falhas/images_to_annotate --debug
python -m defect_detection.train --data datasets/falhas.yaml --model yolo26n.pt --epochs 100 --imgsz 960 --batch 8
python -m defect_detection.predict --weights runs/defects/train/weights/best.pt --source tendence --rectify --debug
```

O que defender:

> A minha etapa usa aprendizado supervisionado para detectar defeitos visuais. Primeiro eu gero/uso recortes retificados da etiqueta, depois anoto as falhas em formato YOLO e treino um modelo para localizar `ranhura`, `amassado` e `mancha`. Usar a etiqueta retificada reduz fundo, escala e perspectiva, deixando o YOLO focar nas falhas.
