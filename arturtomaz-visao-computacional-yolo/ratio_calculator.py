#!/usr/bin/env python3
"""
ratio_calculator.py
===================
Calcula a proporção largura:altura de etiquetas metálicas.

1. Template: detecta os 4 lados retos e calcula o ratio pela intersecção das linhas.
2. Fotos: mesma abordagem + correção de perspectiva via reconstrução 3D.

A chave: os cantos arredondados da etiqueta fazem com que approxPolyDP coloque
vértices nas curvas. Em vez disso, usamos Hough Lines para detectar os 4 lados
retos e calculamos os cantos como intersecção dessas retas.

Uso:
    python ratio_calculator.py <pasta_da_etiqueta>
    Exemplo: python ratio_calculator.py indialar_moveis
"""

import sys
import os
import glob
import math
import cv2
import numpy as np


# =============================================================================
#  TEMPLATE
# =============================================================================

def calculate_template_ratio(template_path: str, debug_dir: str = None) -> dict:
    """Calcula o ratio do template usando detecção de linhas."""
    img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar: {template_path}")
    
    h, w = img.shape[:2]
    pixel_ratio = w / h
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # O fundo dos templates é branco. Invertemos para capturar o objeto.
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Limpa as bordas (5px) para evitar contornos devido a artefatos de exportação
    mask[0:5, :] = 0
    mask[-5:, :] = 0
    mask[:, 0:5] = 0
    mask[:, -5:] = 0
        
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    corners = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        corners = find_true_corners(largest, gray)
    
    if corners is not None:
        tl, tr, br, bl = corners
        line_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        line_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        ratio = line_w / line_h if line_h > 0 else pixel_ratio
        result = {"width": w, "height": h, "ratio": ratio, 
                  "line_w": line_w, "line_h": line_h,
                  "method": "line_intersection", "corners": corners}
    else:
        line_w, line_h = w, h
        result = {"width": w, "height": h, "ratio": pixel_ratio, "method": "pixel"}
        
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        if len(img.shape) == 3 and img.shape[2] == 4:
            debug_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 2:
            debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            debug_img = img.copy()
            
        if corners is not None:
            pts_int = corners.astype(int)
            labels = ["TL", "TR", "BR", "BL"]
            for i in range(4):
                cv2.line(debug_img, tuple(pts_int[i]), tuple(pts_int[(i+1)%4]), (0, 255, 0), 3)
                cv2.circle(debug_img, tuple(pts_int[i]), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, labels[i], tuple(pts_int[i] + np.array([10, -15])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                           
        cv2.putText(debug_img, f"W: {line_w:.1f}px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(debug_img, f"H: {line_h:.1f}px", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        basename = os.path.splitext(os.path.basename(template_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}.jpg"), debug_img)
        
    return result


# =============================================================================
#  ENCONTRAR CANTOS VERDADEIROS VIA HOUGH LINES + FITLINE
# =============================================================================

def find_true_corners(contour: np.ndarray, gray_img: np.ndarray = None) -> np.ndarray | None:
    """
    Encontra os 4 cantos verdadeiros de um retângulo com cantos arredondados.
    
    Abordagem:
    1. Obter o minAreaRect do contorno (dá o ângulo e centro aproximados)
    2. Usar fitLine nos segmentos retos do contorno (excluindo cantos)
    3. Calcular intersecções
    
    A ideia para separar os lados dos cantos: os pontos do contorno que estão
    na região central de cada lado (longe dos cantos) são os que definem
    a reta de cada lado.
    """
    cnt_pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(cnt_pts)
    
    if n < 8:
        return None
    
    # Obter o retângulo orientado que melhor ajusta o contorno
    rect = cv2.minAreaRect(contour)
    center = np.array(rect[0])
    size = rect[1]  # (w, h) — NÃO é garantido qual é largura/altura
    angle = rect[2]  # graus
    
    # boxPoints nos dá os 4 cantos do minAreaRect
    box = cv2.boxPoints(rect)
    box = order_points(box.astype(np.float32))
    # box agora é [tl, tr, br, bl]
    
    # Para cada lado do retângulo, selecionar os pontos do contorno que
    # estão próximos da reta desse lado e longe dos cantos.
    # Definir cada lado pela reta entre dois cantos adjacentes do box.
    
    sides = [
        (box[0], box[1], "top"),
        (box[1], box[2], "right"),
        (box[2], box[3], "bottom"),
        (box[3], box[0], "left"),
    ]
    
    fitted_lines = []
    
    for p_start, p_end, side_name in sides:
        # Vetor do lado
        side_vec = p_end - p_start
        side_len = np.linalg.norm(side_vec)
        if side_len < 1:
            return None
        side_dir = side_vec / side_len
        side_normal = np.array([-side_dir[1], side_dir[0]])
        
        # Para cada ponto do contorno:
        # - Projetar no lado para ver a posição (0 a side_len)
        # - Calcular distância perpendicular ao lado
        projections = []
        for pt in cnt_pts:
            v = pt - p_start
            proj = np.dot(v, side_dir)  # posição ao longo do lado
            dist = abs(np.dot(v, side_normal))  # distância ao lado
            projections.append((proj, dist, pt))
        
        # Filtrar: pontos que estão:
        # - Dentro de 15% a 85% do comprimento do lado (centro, evitando cantos)
        # - Com distância perpendicular < 15% do comprimento do lado
        # Margem adaptativa: lados curtos precisam de margem menor
        margin = 0.15 * side_len
        max_dist = max(0.15 * side_len, 10.0)  # mínimo 10px
        
        side_points = []
        for proj, dist, pt in projections:
            if margin < proj < (side_len - margin) and dist < max_dist:
                side_points.append(pt)
        
        if len(side_points) < 5:
            # Relaxar critérios
            margin = 0.05 * side_len
            max_dist = max(0.25 * side_len, 15.0)
            side_points = []
            for proj, dist, pt in projections:
                if margin < proj < (side_len - margin) and dist < max_dist:
                    side_points.append(pt)
        
        if len(side_points) < 2:
            return None
        
        pts_array = np.array(side_points, dtype=np.float32).reshape(-1, 1, 2)
        line = cv2.fitLine(pts_array, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        fitted_lines.append((vx, vy, float(x0), float(y0), side_name))
    
    if len(fitted_lines) != 4:
        return None
    
    # Calcular as 4 intersecções: top∩left=TL, top∩right=TR, bottom∩right=BR, bottom∩left=BL
    def line_intersect(l1, l2):
        vx1, vy1, x01, y01 = l1[:4]
        vx2, vy2, x02, y02 = l2[:4]
        denom = vx1 * vy2 - vy1 * vx2
        if abs(denom) < 1e-10:
            return None
        t = ((x02 - x01) * vy2 - (y02 - y01) * vx2) / denom
        x = x01 + t * vx1
        y = y01 + t * vy1
        return np.array([x, y], dtype=np.float32)
    
    # sides order: 0=top, 1=right, 2=bottom, 3=left
    tl = line_intersect(fitted_lines[0], fitted_lines[3])  # top ∩ left
    tr = line_intersect(fitted_lines[0], fitted_lines[1])  # top ∩ right
    br = line_intersect(fitted_lines[1], fitted_lines[2])  # right ∩ bottom
    bl = line_intersect(fitted_lines[2], fitted_lines[3])  # bottom ∩ left
    
    if any(p is None for p in [tl, tr, br, bl]):
        return None
    
    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    
    # Sanity check: a área dos cantos deve ser próxima da área do contorno
    area_corners = cv2.contourArea(corners.reshape(4, 1, 2).astype(np.int32))
    area_contour = cv2.contourArea(contour)
    if area_contour > 0:
        ratio_area = area_corners / area_contour
        # Os cantos verdadeiros devem dar uma área um pouco maior (por causa dos arredondamentos)
        if ratio_area < 0.8 or ratio_area > 1.5:
            return None
    
    return corners


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# =============================================================================
#  CORREÇÃO DE PERSPECTIVA (Zhang 3D)
# =============================================================================

def line_intersection_homogeneous(p1, p2, p3, p4):
    """Intersecção via coordenadas homogêneas."""
    l1 = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    l2 = np.cross([p3[0], p3[1], 1.0], [p4[0], p4[1], 1.0])
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10:
        return None
    return np.array([pt[0] / pt[2], pt[1] / pt[2]])


def compute_aspect_ratio(corners: np.ndarray, img_w: int, img_h: int) -> dict:
    """Calcula o aspect ratio com múltiplos métodos."""
    tl, tr, br, bl = corners
    u0, v0 = img_w / 2.0, img_h / 2.0
    
    vp_h = line_intersection_homogeneous(tl, tr, bl, br)
    vp_v = line_intersection_homogeneous(tl, bl, tr, br)
    
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    simple_ratio = (w_top + w_bot) / (h_left + h_right)
    
    zhang_ratio = None
    focal_length = None
    
    if vp_h is not None and vp_v is not None:
        vp_h_c = vp_h - np.array([u0, v0])
        vp_v_c = vp_v - np.array([u0, v0])
        f_sq = -(vp_h_c[0] * vp_v_c[0] + vp_h_c[1] * vp_v_c[1])
        
        if f_sq > 0:
            focal_length = math.sqrt(f_sq)
            
            def to_3d(p):
                return np.array([p[0] - u0, p[1] - v0, focal_length])
            
            d1 = to_3d(vp_h); d1 /= np.linalg.norm(d1)
            d2 = to_3d(vp_v); d2 /= np.linalg.norm(d2)
            normal = np.cross(d1, d2)
            normal /= np.linalg.norm(normal)
            
            pts_3d = []
            valid = True
            for m in corners:
                ray = to_3d(m)
                denom = np.dot(normal, ray)
                if abs(denom) < 1e-12:
                    valid = False; break
                pts_3d.append(ray / denom)
            
            if valid:
                p1, p2, p3, p4 = pts_3d
                w3d = (np.linalg.norm(p2-p1) + np.linalg.norm(p3-p4)) / 2
                h3d = (np.linalg.norm(p4-p1) + np.linalg.norm(p3-p2)) / 2
                if h3d > 1e-10:
                    candidate = w3d / h3d
                    if 1.0 < candidate < 10.0:
                        zhang_ratio = candidate
    
    if zhang_ratio is not None:
        best_ratio = zhang_ratio
        best_method = "Zhang 3D"
    else:
        best_ratio = simple_ratio
        best_method = "simples"
    
    return {
        "simple_ratio": simple_ratio,
        "zhang_ratio": zhang_ratio, "best_ratio": best_ratio,
        "best_method": best_method, "focal_length": focal_length,
        "vp_horizontal": vp_h, "vp_vertical": vp_v,
        "width_top": w_top, "width_bottom": w_bot,
        "height_left": h_left, "height_right": h_right,
    }


# =============================================================================
#  DETECCAO DE COR ALTERADA
# =============================================================================

def warp_label(image: np.ndarray, corners: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Aplica perspectiva para alinhar a etiqueta em uma vista frontal."""
    src = corners.astype(np.float32)
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def _estimate_warp_size(corners: np.ndarray, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    """Estima um tamanho retificado estável a partir dos cantos detectados."""
    tl, tr, br, bl = corners
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0))
    width = max(width, 32, int(fallback_w))
    height = max(height, 32, int(fallback_h))
    return width, height


def _build_template_masks(template_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa a base metálica das regiões impressas usando o template como referência.
    """
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # A base metálica muda suavemente; texto, bordas e logo aparecem como
    # desvios locais mais escuros/saturados em relação a uma versão suavizada.
    blur_size = max(21, (min(template_bgr.shape[:2]) // 8) | 1)
    smooth_bgr = cv2.GaussianBlur(template_bgr, (blur_size, blur_size), 0)
    smooth_gray = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2GRAY)
    smooth_hsv = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2HSV)

    _, s, _ = cv2.split(hsv)
    _, smooth_s, _ = cv2.split(smooth_hsv)

    detail_gray = cv2.absdiff(gray, smooth_gray)
    detail_sat = cv2.absdiff(s, smooth_s)
    dark_mask = gray < max(110, int(np.percentile(gray, 28)))
    detail_mask = detail_gray > max(12, int(np.percentile(detail_gray, 82)))
    color_mask = (s > max(35, int(np.percentile(s, 80)))) | (detail_sat > max(10, int(np.percentile(detail_sat, 82))))

    print_mask = ((dark_mask & detail_mask) | color_mask).astype(np.uint8) * 255
    print_mask = cv2.morphologyEx(print_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    print_mask = cv2.dilate(print_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    label_mask = np.ones_like(print_mask, dtype=np.uint8) * 255
    base_mask = cv2.bitwise_and(cv2.bitwise_not(print_mask), label_mask)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

    return label_mask, base_mask, print_mask


def _compute_mask_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Média por canal dentro da máscara."""
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return None
    return np.mean(pixels.astype(np.float32), axis=0)


def _normalize_photo_color(photo_bgr: np.ndarray, template_bgr: np.ndarray, base_mask: np.ndarray) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """
    Usa a base metálica para estimar iluminação/cast de cor e normaliza a foto.
    """
    photo_lab = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    template_lab = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    photo_base_mean = _compute_mask_mean(photo_lab, base_mask)
    template_base_mean = _compute_mask_mean(template_lab, base_mask)
    if photo_base_mean is None or template_base_mean is None:
        return None, None

    # Corrige brilho + cast cromático usando a base metálica como referência.
    delta = template_base_mean - photo_base_mean
    normalized_lab = photo_lab + delta
    normalized_lab[:, :, 0] = np.clip(normalized_lab[:, :, 0], 0, 255)
    normalized_lab[:, :, 1] = np.clip(normalized_lab[:, :, 1], 0, 255)
    normalized_lab[:, :, 2] = np.clip(normalized_lab[:, :, 2], 0, 255)

    normalized_bgr = cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    stats = {
        "photo_base_lab": photo_base_mean,
        "template_base_lab": template_base_mean,
        "delta_lab": delta,
    }
    return normalized_bgr, stats


def prepare_template_color_reference(template_path: str, debug_dir: str = None) -> dict | None:
    """Cria a referência de cor do template já retificada e mascarada."""
    img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    is_fake = "fake_template" in os.path.basename(template_path).lower()

    if is_fake:
        corners, (_h, _w), _ = detect_label_in_photo(template_path, debug_dir)
        if corners is None:
            return None
        out_w, out_h = _estimate_warp_size(corners, 400, 250)
        rectified = warp_label(img, corners, out_w, out_h)
    else:
        result = calculate_template_ratio(template_path, debug_dir)
        corners = result.get("corners")
        if corners is None:
            corners = np.array([
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ], dtype=np.float32)
        out_w, out_h = _estimate_warp_size(corners, w, h)
        rectified = warp_label(img, corners, out_w, out_h)

    label_mask, base_mask, print_mask = _build_template_masks(rectified)
    reference = {
        "path": template_path,
        "image": rectified,
        "width": out_w,
        "height": out_h,
        "label_mask": label_mask,
        "base_mask": base_mask,
        "print_mask": print_mask,
    }

    if debug_dir:
        overlay = rectified.copy()
        overlay[base_mask > 0] = cv2.addWeighted(overlay, 0.5, np.full_like(overlay, (0, 255, 0)), 0.5, 0)[base_mask > 0]
        overlay[print_mask > 0] = cv2.addWeighted(overlay, 0.5, np.full_like(overlay, (0, 0, 255)), 0.5, 0)[print_mask > 0]
        cv2.imwrite(os.path.join(debug_dir, "debug_template_color_reference.jpg"), overlay)

    return reference


def detect_color_change(photo_path: str, corners: np.ndarray, template_ref: dict, debug_dir: str = None) -> dict | None:
    """
    Compara a cor da foto com o template após compensar a iluminação pela base metálica.
    """
    img = cv2.imread(photo_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    warped = warp_label(img, corners, template_ref["width"], template_ref["height"])
    normalized, norm_stats = _normalize_photo_color(warped, template_ref["image"], template_ref["base_mask"])
    if normalized is None:
        return None

    # Suaviza levemente para reduzir ruído de textura/antialiasing.
    template_smooth = cv2.GaussianBlur(template_ref["image"], (5, 5), 0)
    normalized_smooth = cv2.GaussianBlur(normalized, (5, 5), 0)

    template_lab = cv2.cvtColor(template_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    normalized_lab = cv2.cvtColor(normalized_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_lab = normalized_lab - template_lab
    delta_e = np.linalg.norm(delta_lab, axis=2)
    delta_chroma = np.linalg.norm(delta_lab[:, :, 1:3], axis=2)

    base_mask = template_ref["base_mask"]
    print_mask = template_ref["print_mask"]
    label_mask = template_ref["label_mask"]
    template_sat = cv2.cvtColor(template_ref["image"], cv2.COLOR_BGR2HSV)[:, :, 1]
    color_print_mask = ((print_mask > 0) & (template_sat > 60))

    mean_base_delta = float(np.mean(delta_chroma[base_mask > 0])) if np.any(base_mask > 0) else 0.0
    mean_print_delta = float(np.mean(delta_chroma[print_mask > 0])) if np.any(print_mask > 0) else 0.0
    mean_label_delta = float(np.mean(delta_chroma[label_mask > 0])) if np.any(label_mask > 0) else 0.0
    max_print_delta = float(np.max(delta_chroma[print_mask > 0])) if np.any(print_mask > 0) else 0.0
    mean_color_print_delta = float(np.mean(delta_chroma[color_print_mask])) if np.any(color_print_mask) else 0.0
    p90_color_print_delta = float(np.percentile(delta_chroma[color_print_mask], 90)) if np.any(color_print_mask) else 0.0
    local_color_component_score = 0.0
    local_color_component_p95 = 0.0
    local_color_component_area = 0

    if np.any(color_print_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(color_print_mask.astype(np.uint8), 8)
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < 150:
                continue
            component_values = delta_chroma[labels == label_idx]
            component_mean = float(np.mean(component_values))
            component_p95 = float(np.percentile(component_values, 95))
            if (component_mean, component_p95, area) > (local_color_component_score, local_color_component_p95, local_color_component_area):
                local_color_component_score = component_mean
                local_color_component_p95 = component_p95
                local_color_component_area = area

    # A base deve ficar parecida após normalização; o que sobra nas regiões
    # impressas é um bom indicador de cor alterada.
    effective_delta = max(0.0, mean_print_delta - 0.6 * mean_base_delta)
    color_changed = (
        effective_delta > 8.0 or
        (mean_print_delta > 10.0 and max_print_delta > 18.0) or
        (mean_color_print_delta > 4.5 and p90_color_print_delta > 8.0) or
        (local_color_component_score > 5.0 and local_color_component_p95 > 7.5 and local_color_component_area >= 150)
    )

    result = {
        "changed": color_changed,
        "effective_delta": effective_delta,
        "mean_base_delta": mean_base_delta,
        "mean_print_delta": mean_print_delta,
        "mean_label_delta": mean_label_delta,
        "max_print_delta": max_print_delta,
        "mean_color_print_delta": mean_color_print_delta,
        "p90_color_print_delta": p90_color_print_delta,
        "local_color_component_score": local_color_component_score,
        "local_color_component_p95": local_color_component_p95,
        "local_color_component_area": local_color_component_area,
        "normalized_image": normalized,
        "warped_image": warped,
        "delta_lab": norm_stats["delta_lab"],
    }

    if debug_dir:
        basename = os.path.splitext(os.path.basename(photo_path))[0]
        heat = np.clip((delta_e / 32.0) * 255.0, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.addWeighted(normalized, 0.65, heat, 0.35, 0)
        cv2.putText(heat, f"print dE={mean_print_delta:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heat, f"base dE={mean_base_delta:.1f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heat, f"color dE={mean_color_print_delta:.1f}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(heat, f"status={'ALTERADA' if color_changed else 'OK'}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}_color_normalized.jpg"), normalized)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}_color_heatmap.jpg"), heat)

    return result


# =============================================================================
#  DETECÇÃO DE CONTORNO NAS FOTOS
# =============================================================================

def _normalize_shadows(gray: np.ndarray) -> np.ndarray:
    """
    Remove sombras e gradientes de iluminação via normalização por divisão.
    Divide a imagem por uma versão muito borrada dela mesma (estimativa do
    fundo/iluminação), resultando em contraste uniforme.
    """
    h, w = gray.shape[:2]
    blur_size = max(w, h) // 3
    if blur_size % 2 == 0:
        blur_size += 1
    blur_size = max(blur_size, 3)
    bg = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def _edge_contrast_score(gray: np.ndarray, contour: np.ndarray, sample_width: int = 5) -> float:
    """
    Avalia a nitidez (contraste) da borda de um contorno.
    Amostra pixels dentro e fora do contorno ao longo do perímetro e calcula
    a diferença média. Sombras suaves retornam score baixo; bordas metálicas
    nítidas retornam score alto.
    Retorna valor entre 0 (sem contraste) e 255 (contraste máximo).
    """
    h, w = gray.shape[:2]
    mask_in  = np.zeros((h, w), dtype=np.uint8)
    mask_out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_in,  [contour], -1, 255, sample_width)
    # "out" = dilatar ainda mais e subtrair
    dilated = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_RECT, (sample_width * 2 + 1,) * 2))
    mask_out = cv2.subtract(dilated, mask_in)
    # Erode a máscara interna para amostrar só a borda interna
    eroded = cv2.erode(mask_in, cv2.getStructuringElement(cv2.MORPH_RECT, (sample_width * 2 + 1,) * 2))
    mask_in = cv2.subtract(mask_in, eroded)
    
    pixels_in  = gray[mask_in  > 0]
    pixels_out = gray[mask_out > 0]
    if len(pixels_in) == 0 or len(pixels_out) == 0:
        return 0.0
    return float(abs(np.mean(pixels_in.astype(np.float32)) - np.mean(pixels_out.astype(np.float32))))


def _find_best_contour(blurred: np.ndarray, h_img: int, w_img: int,
                       target_ratio: float = None, ratio_tol: float = 0.40,
                       allow_perspective_fallback: bool = False):
    """
    Busca o melhor contorno retangular na imagem pré-processada.

    Melhorias vs. versão anterior:
    - Penaliza contornos muito grandes (> 60 % da imagem) — evita pegar o
      gradiente/sombra da parede que forma um retângulo maior que a placa.
    - Valida o contraste real da borda: sombras suaves têm contraste baixo;
      bordas metálicas têm contraste alto.  Contornos com contraste < limiar
      são descartados.
    - Se target_ratio for fornecido (vindo do template), descarta contornos
      cujo aspect ratio diverge mais que ratio_tol (relativo).

    Retorna (contour, score) ou (None, 0).
    """
    best_contour = None
    best_score   = 0
    img_area = h_img * w_img

    def _evaluate_cnt(cnt):
        """Retorna score ou 0 se o contorno for inválido."""
        area = cv2.contourArea(cnt)
        # --- área mínima e máxima ---
        if area < img_area * 0.01 or area > img_area * 0.75:
            return 0

        x, y, w, h = cv2.boundingRect(cnt)
        if w > w_img * 0.95 and h > h_img * 0.95:
            return 0

        # --- retangularidade ---
        rect = cv2.minAreaRect(cnt)
        box  = cv2.boxPoints(rect)
        box_area = cv2.contourArea(box)
        rectangularity = area / box_area if box_area > 0 else 0
        effective_rectangularity = rectangularity
        if allow_perspective_fallback:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            effective_rectangularity = max(rectangularity, hull_area / box_area if box_area > 0 else 0)

        if effective_rectangularity < 0.75:
            return 0

        # --- aspect ratio básico ---
        wr, hr = rect[1]
        if wr == 0 or hr == 0:
            return 0
        ar = max(wr, hr) / min(wr, hr)
        if ar < 1.1 or ar > 8.0:
            return 0

        # --- constraint de ratio pelo template ---
        ratio_penalty = 1.0
        if target_ratio is not None:
            # Em fotos com forte perspectiva, o aspect ratio aparente pode
            # distorcer bastante. Em vez de descartar cedo, penalizamos.
            # ar pode ser W/H ou H/W; testamos ambos
            ar_direct  = max(wr, hr) / min(wr, hr)
            ar_flipped = min(wr, hr) / max(wr, hr) if max(wr, hr) > 0 else 0
            best_ar = ar_direct if abs(ar_direct - target_ratio) < abs(ar_flipped - target_ratio) else ar_flipped
            rel_diff = abs(best_ar - target_ratio) / target_ratio
            if allow_perspective_fallback:
                if rel_diff > max(ratio_tol * 4, 2.0):
                    return 0
                if rel_diff > ratio_tol:
                    ratio_penalty = 1.0 / (1.0 + 1.5 * (rel_diff - ratio_tol))
            elif rel_diff > ratio_tol:
                return 0

        # --- penalidade por tamanho excessivo ---
        # Contornos maiores que 55% da imagem recebem penalidade crescente.
        area_frac = area / img_area
        size_penalty = max(0.0, 1.0 - ((area_frac - 0.55) / 0.20)) if area_frac > 0.55 else 1.0

        # --- contraste de borda ---
        contrast = _edge_contrast_score(blurred, cnt, sample_width=4)
        # Limiar: sombras de parede tipicamente têm contraste < 8; placas metálicas > 15.
        if contrast < 6:
            return 0
        contrast_factor = min(contrast / 40.0, 1.0)  # normaliza; satura em 40

        return area * effective_rectangularity * size_penalty * contrast_factor * ratio_penalty

    all_contours = []

    for low, high in [(30, 100), (50, 150), (20, 80), (40, 120), (80, 160), (100, 200), (120, 240)]:
        edges = cv2.Canny(blurred, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_contours.extend(contours)

    # Estratégias de threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for edge_img in [
        cv2.morphologyEx(cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv2.THRESH_BINARY_INV, 11, 2), cv2.MORPH_CLOSE, kernel, iterations=2),
        cv2.morphologyEx(cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                         cv2.MORPH_CLOSE, kernel, iterations=2),
        cv2.morphologyEx(cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
                         cv2.MORPH_CLOSE, kernel, iterations=1),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ]:
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_contours.extend(contours)

    for cnt in all_contours:
        score = _evaluate_cnt(cnt)
        if score > best_score:
            best_score   = score
            best_contour = cnt

    return best_contour, best_score


def detect_label_in_photo(photo_path: str, debug_dir: str = None,
                          target_ratio: float = None, ratio_tol: float = 0.40):
    """
    Detecta a etiqueta na foto e retorna os cantos corrigidos via
    intersecção de linhas dos lados retos.

    Parâmetros
    ----------
    photo_path   : caminho da imagem.
    debug_dir    : diretório para salvar imagens de debug.
    target_ratio : aspect ratio esperado (W/H), vindo do template.
                   Quando fornecido, contornos com ratio incompatível são
                   descartados em _find_best_contour.
    ratio_tol    : tolerância relativa para o filtro de ratio (padrão 40 %).

    Retorna: (cantos_corrigidos, (h_img, w_img), best_contour)
    """
    img = cv2.imread(photo_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar: {photo_path}")

    original = img.copy()
    h_img, w_img = img.shape[:2]

    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _evaluate_corners(cnt, gray):
        """Retorna (corners, quality_score). Quality mais alto = melhor."""
        if cnt is None:
            return None, -1
        corners = find_true_corners(cnt, gray)
        if corners is None:
            return None, 0
        tl, tr, br, bl = corners
        w_top   = np.linalg.norm(tr - tl)
        w_bot   = np.linalg.norm(br - bl)
        h_left  = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        w_diff = abs(w_top - w_bot) / max(w_top, w_bot) if max(w_top, w_bot) > 0 else 1
        h_diff = abs(h_left - h_right) / max(h_left, h_right) if max(h_left, h_right) > 0 else 1
        corner_area = cv2.contourArea(corners.reshape(4, 1, 2).astype(np.int32))
        cnt_area    = cv2.contourArea(cnt)
        area_ratio  = corner_area / cnt_area if cnt_area > 0 else 0
        max_asym    = max(w_diff, h_diff)
        symmetry    = (1.0 - max_asym) ** 3
        quality     = cnt_area * symmetry * min(area_ratio, 1.0)
        return corners, quality

    # Pipeline 1: original
    gray1    = clahe.apply(gray_raw)
    blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    cnt1, _  = _find_best_contour(blurred1, h_img, w_img,
                                   target_ratio=target_ratio, ratio_tol=ratio_tol)
    corners1, q1 = _evaluate_corners(cnt1, gray1)

    if corners1 is None:
        cnt1_relaxed, _ = _find_best_contour(
            blurred1, h_img, w_img,
            target_ratio=target_ratio, ratio_tol=ratio_tol,
            allow_perspective_fallback=True
        )
        corners1_relaxed, q1_relaxed = _evaluate_corners(cnt1_relaxed, gray1)
        if q1_relaxed > q1:
            cnt1, corners1, q1 = cnt1_relaxed, corners1_relaxed, q1_relaxed

    best_contour, best_corners, best_quality, best_gray = cnt1, corners1, q1, gray1

    # Pipelines 2+: normalização de sombras
    for blur_div in [5, 3, 2]:
        blur_size = max(w_img, h_img) // blur_div
        if blur_size % 2 == 0:
            blur_size += 1
        blur_size = max(blur_size, 3)
        bg        = cv2.GaussianBlur(gray_raw, (blur_size, blur_size), 0)
        gray_norm = cv2.divide(gray_raw, bg, scale=255)
        gray_n    = clahe.apply(gray_norm)
        blurred_n = cv2.GaussianBlur(gray_n, (5, 5), 0)
        cnt_n, _  = _find_best_contour(blurred_n, h_img, w_img,
                                        target_ratio=target_ratio, ratio_tol=ratio_tol)
        corners_n, q_n = _evaluate_corners(cnt_n, gray_n)
        if corners_n is None:
            cnt_n_relaxed, _ = _find_best_contour(
                blurred_n, h_img, w_img,
                target_ratio=target_ratio, ratio_tol=ratio_tol,
                allow_perspective_fallback=True
            )
            corners_n_relaxed, q_n_relaxed = _evaluate_corners(cnt_n_relaxed, gray_n)
            if q_n_relaxed > q_n:
                cnt_n, corners_n, q_n = cnt_n_relaxed, corners_n_relaxed, q_n_relaxed
        if q_n > best_quality:
            best_contour, best_corners, best_quality, best_gray = cnt_n, corners_n, q_n, gray_n

    true_corners = best_corners

    if best_contour is None:
        return None, (h_img, w_img), None

    # Debug
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = original.copy()

        cv2.drawContours(debug_img, [best_contour], -1, (128, 128, 128), 1)

        if true_corners is not None:
            pts_int = true_corners.astype(int)
            labels  = ["TL", "TR", "BR", "BL"]
            for i in range(4):
                cv2.line(debug_img, tuple(pts_int[i]), tuple(pts_int[(i+1)%4]), (0, 255, 0), 3)
                cv2.circle(debug_img, tuple(pts_int[i]), 8, (0, 0, 255), -1)
                cv2.putText(debug_img, labels[i], tuple(pts_int[i] + np.array([10, -15])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        peri   = cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, 0.02 * peri, True)
        for p in approx.reshape(-1, 2):
            cv2.circle(debug_img, tuple(p), 5, (0, 165, 255), -1)

        basename = os.path.splitext(os.path.basename(photo_path))[0]
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}.jpg"), debug_img)

    return true_corners, (h_img, w_img), best_contour


# =============================================================================
#  MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Uso: python ratio_calculator.py <pasta_da_etiqueta>")
        sys.exit(1)
    
    label_dir = sys.argv[1]
    if not os.path.isdir(label_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        label_dir = os.path.join(script_dir, label_dir)
    if not os.path.isdir(label_dir):
        print(f"Erro: diretório '{label_dir}' não encontrado.")
        sys.exit(1)
    
    debug_dir = os.path.join(label_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, "log.txt")
    f_log = open(log_path, "w", encoding="utf-8")
    
    print("=" * 70)
    print("  CALCULADOR DE PROPORÇÃO — INTERSECÇÃO DE LINHAS DOS LADOS")
    print("=" * 70)
    print(f"\nDiretório: {label_dir}\n")
    
    # =========================================================================
    # 1. TEMPLATE
    # =========================================================================
    template_ratio = None
    template_color_ref = None
    
    template_files = glob.glob(os.path.join(label_dir, "template.*"))
    template_files = [f for f in template_files if "fake_template" not in os.path.basename(f).lower()]
    fake_template_files = glob.glob(os.path.join(label_dir, "fake_template.*"))
    
    if template_files:
        template_path = template_files[0]
        t = calculate_template_ratio(template_path, debug_dir)
        template_ratio = t["ratio"]
        
        print("─" * 70)
        print("  TEMPLATE")
        print("─" * 70)
        print(f"  Arquivo:    {os.path.basename(template_path)}")
        print(f"  Pixels:     {t['width']} x {t['height']} px")
        print(f"  Ratio:      {t['ratio']:.4f}  ({t['method']})")
        if 'line_w' in t:
            print(f"  Dim linhas: {t['line_w']:.1f} x {t['line_h']:.1f}")
        from math import gcd
        g = gcd(t['width'], t['height'])
        print(f"  Simplif.:   {t['width']//g}:{t['height']//g}")
        print()
        template_color_ref = prepare_template_color_reference(template_path, debug_dir)
        
        f_log.write(f"[template]\n")
        f_log.write(f"dimensão: {t['width']}x{t['height']}\n")
        f_log.write(f"tecnica: {t['method']}\n\n")
    elif fake_template_files:
        template_path = fake_template_files[0]
        true_corners, img_dims, _ = detect_label_in_photo(template_path, debug_dir)
        if true_corners is not None:
            h_img, w_img = img_dims
            t = compute_aspect_ratio(true_corners, w_img, h_img)
            template_ratio = t['best_ratio']
            
            print("─" * 70)
            print("  FAKE TEMPLATE")
            print("─" * 70)
            print(f"  Arquivo:    {os.path.basename(template_path)}")
            print(f"  Ratio:      {template_ratio:.4f}  ({t['best_method']})")
            print()
            template_color_ref = prepare_template_color_reference(template_path, debug_dir)
            
            f_log.write(f"[fake template]\n")
            f_log.write(f"dimensão: ratio {template_ratio:.4f} ({w_img}x{h_img})\n")
            f_log.write(f"tecnica: {t['best_method']}\n\n")
        else:
            print(f"⚠ Não foi possível detectar a etiqueta no fake_template: {os.path.basename(template_path)}")
    else:
        print("⚠ Template não encontrado!")
    
    # =========================================================================
    # 2. FOTOS
    # =========================================================================
    photo_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    photos = []
    for pattern in photo_patterns:
        photos.extend(glob.glob(os.path.join(label_dir, pattern)))
        
    # Remover duplicatas e filtrar arquivos com 'template' no nome
    photos = sorted(list(set([p for p in photos if "template" not in os.path.basename(p).lower()])))
    
    if not photos:
        print("⚠ Nenhuma foto encontrada!"); sys.exit(0)
    
    print("─" * 70)
    print("  FOTOS (Intersecção de Linhas + Correção de Perspectiva)")
    print("─" * 70)
    
    all_results = []
    
    for photo_path in photos:
        basename = os.path.basename(photo_path)
        base_name_no_ext = os.path.splitext(basename)[0]
        print(f"\n  📷 {basename}")
        
        true_corners, (h_img, w_img), best_contour = detect_label_in_photo(
            photo_path, debug_dir, target_ratio=template_ratio)
        
        if true_corners is None:
            print(f"     ❌ Não foi possível detectar/ajustar linhas")
            continue
        
        result = compute_aspect_ratio(true_corners, w_img, h_img)
        all_results.append(result)
        color_result = detect_color_change(photo_path, true_corners, template_color_ref, debug_dir) if template_color_ref else None
        
        # LOG
        diff = abs(result['best_ratio'] - template_ratio) / template_ratio * 100 if template_ratio else 0
        f_log.write(f"[{basename}]\n")
        f_log.write(f"Simples: {result['simple_ratio']:.4f}\n")
        if result['zhang_ratio'] is not None:
            f_log.write(f"Zhang: {result['zhang_ratio']:.4f}\n")
        else:
            f_log.write(f"Zhang: N/A\n")
        f_log.write(f"Melhor: {result['best_method']} {result['best_ratio']:.4f}\n")
        if color_result is not None:
            f_log.write(f"Cor alterada: {'SIM' if color_result['changed'] else 'NAO'}\n")
            f_log.write(f"dE base: {color_result['mean_base_delta']:.2f}\n")
            f_log.write(f"dE impresso: {color_result['mean_print_delta']:.2f}\n")
            f_log.write(f"dE impresso colorido: {color_result['mean_color_print_delta']:.2f}\n")
            f_log.write(f"dE p90 colorido: {color_result['p90_color_print_delta']:.2f}\n")
            f_log.write(f"dE comp. local colorido: {color_result['local_color_component_score']:.2f}\n")
            f_log.write(f"dE p95 comp. local: {color_result['local_color_component_p95']:.2f}\n")
            f_log.write(f"Area comp. local: {color_result['local_color_component_area']}\n")
            f_log.write(f"dE efetivo: {color_result['effective_delta']:.2f}\n")
        f_log.write(f"Diferença: {diff:.1f}%\n\n")
        
        # DEBUGS ADICIONAIS
        img_orig = cv2.imread(photo_path)
        pts_int = true_corners.astype(int)
        
        # debug_{foto}_simples.jpg
        img_simples = img_orig.copy()
        cv2.polylines(img_simples, [pts_int], True, (255, 0, 0), 3) # Blue
        cv2.putText(img_simples, f"Simples Ratio: {result['simple_ratio']:.4f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{base_name_no_ext}_simples.jpg"), img_simples)
        
        # debug_{foto}_zhang.jpg
        if result['zhang_ratio'] is not None:
            img_zhang = img_orig.copy()
            cv2.polylines(img_zhang, [pts_int], True, (0, 0, 255), 3) # Red
            cv2.putText(img_zhang, f"Zhang Ratio: {result['zhang_ratio']:.4f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(debug_dir, f"debug_{base_name_no_ext}_zhang.jpg"), img_zhang)
            
        # debug_{foto}_findContours.jpg
        img_fc = img_orig.copy()
        cv2.drawContours(img_fc, [best_contour], -1, (0, 255, 255), 2)
        cv2.putText(img_fc, f"findContours (raw)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{base_name_no_ext}_findContours.jpg"), img_fc)
        
        # debug_{foto}_contourArea.jpg
        img_ca = img_orig.copy()
        cv2.drawContours(img_ca, [best_contour], -1, (0, 255, 0), -1) # Filled
        area = cv2.contourArea(best_contour)
        cv2.putText(img_ca, f"contourArea: {area:.0f} px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{base_name_no_ext}_contourArea.jpg"), img_ca)
        
        print(f"     Imagem: {w_img}x{h_img} px")
        print(f"     Larg: top={result['width_top']:.1f}  bot={result['width_bottom']:.1f} px")
        print(f"     Alt:  left={result['height_left']:.1f}  right={result['height_right']:.1f} px")
        print(f"     ┌─ Simples:     {result['simple_ratio']:.4f}")
        if result['zhang_ratio'] is not None:
            print(f"     ├─ Zhang 3D:    {result['zhang_ratio']:.4f}  (f={result['focal_length']:.0f}px)")
        else:
            print(f"     ├─ Zhang 3D:    N/A")
        print(f"     └─ ★ MELHOR:     {result['best_ratio']:.4f}  ({result['best_method']})")
    
    # =========================================================================
    # RESUMO
    # =========================================================================
    if all_results:
        print()
        print("─" * 70)
        print("  RESUMO")
        print("─" * 70)
        
        if template_ratio is not None:
            print(f"\n  Template ratio: {template_ratio:.4f}")
        
        def summarize(name, values):
            if not values: return
            avg = sum(values) / len(values)
            std = (sum((v - avg)**2 for v in values) / len(values)) ** 0.5
            line = f"    Média: {avg:.4f}  σ: {std:.4f}  [{min(values):.4f}—{max(values):.4f}]"
            if template_ratio:
                diff = abs(avg - template_ratio) / template_ratio * 100
                line += f"  Δ={diff:.1f}%"
            print(f"\n  {name}:")
            print(line)
        
        summarize("Simples", [r["simple_ratio"] for r in all_results])
        zhang_vals = [r["zhang_ratio"] for r in all_results if r["zhang_ratio"] is not None]
        if zhang_vals:
            summarize("Zhang 3D", zhang_vals)
        summarize("★ Melhor", [r["best_ratio"] for r in all_results])
        
        print(f"\n  Debug em: {debug_dir}/")
    
    f_log.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
