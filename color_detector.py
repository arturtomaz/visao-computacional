import os
import sys
import glob
import cv2
import numpy as np


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
    color_mask = (s > max(75, int(np.percentile(s, 80)))) | (detail_sat > max(10, int(np.percentile(detail_sat, 82))))

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


def prepare_template_color_reference(template_path: str, corners: np.ndarray = None, debug_dir: str = None) -> dict | None:
    """Cria a referência de cor do template já retificada e mascarada."""
    img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    is_fake = "fake_template" in os.path.basename(template_path).lower()

    if corners is None:
        corners = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

    if is_fake:
        out_w, out_h = _estimate_warp_size(corners, 400, 250)
    else:
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
    color_print_mask = ((print_mask > 0) & (template_sat > 100))

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
        cv2.putText(heat, f"status={'ALTERADA' if color_changed else 'OK'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}_color_normalized.jpg"), normalized)
        cv2.imwrite(os.path.join(debug_dir, f"debug_{basename}_color_heatmap.jpg"), heat)

    return result


def main():
    if len(sys.argv) < 2:
        print("Uso: python color_detector.py <pasta_da_etiqueta>")
        sys.exit(1)
        
    label_dir = sys.argv[1]
    if not os.path.isdir(label_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        label_dir = os.path.join(script_dir, label_dir)
    if not os.path.isdir(label_dir):
        print(f"Erro: diretório '{label_dir}' não encontrado.")
        sys.exit(1)
        
    debug_dir = os.path.join(label_dir, "debug_color")
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, "color_log.txt")
    f_log = open(log_path, "w", encoding="utf-8")
    
    print("=" * 70)
    print("  DETECTOR DE COR ALTERADA")
    print("=" * 70)
    print(f"\nDiretório: {label_dir}\n")
    
    from ratio_calculator import calculate_template_ratio, detect_label_in_photo, compute_aspect_ratio
    
    template_ratio = None
    template_color_ref = None
    
    template_files = glob.glob(os.path.join(label_dir, "template.*"))
    template_files = [f for f in template_files if "fake_template" not in os.path.basename(f).lower()]
    fake_template_files = glob.glob(os.path.join(label_dir, "fake_template.*"))
    
    if template_files:
        template_path = template_files[0]
        t = calculate_template_ratio(template_path, debug_dir)
        template_ratio = t["ratio"]
        template_color_ref = prepare_template_color_reference(template_path, t.get("corners"), debug_dir)
    elif fake_template_files:
        template_path = fake_template_files[0]
        true_corners, img_dims, _ = detect_label_in_photo(template_path, debug_dir)
        if true_corners is not None:
            h_img, w_img = img_dims
            t = compute_aspect_ratio(true_corners, w_img, h_img)
            template_ratio = t['best_ratio']
            template_color_ref = prepare_template_color_reference(template_path, true_corners, debug_dir)
            
    if not template_color_ref:
        print("⚠ Template não encontrado ou inválido para cor!")
        sys.exit(1)
        
    photo_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    photos = []
    for pattern in photo_patterns:
        photos.extend(glob.glob(os.path.join(label_dir, pattern)))
        
    photos = sorted(list(set([p for p in photos if "template" not in os.path.basename(p).lower()])))
    
    if not photos:
        print("⚠ Nenhuma foto encontrada!")
        sys.exit(0)
        
    for photo_path in photos:
        basename = os.path.basename(photo_path)
        print(f"\n  📷 {basename}")
        
        true_corners, _, _ = detect_label_in_photo(photo_path, debug_dir, target_ratio=template_ratio)
        if true_corners is None:
            print(f"     ❌ Não foi possível detectar a etiqueta")
            continue
            
        color_result = detect_color_change(photo_path, true_corners, template_color_ref, debug_dir)
        if color_result is not None:
            f_log.write(f"[{basename}]\n")
            f_log.write(f"Cor alterada: {'SIM' if color_result['changed'] else 'NAO'}\n")
            f_log.write(f"dE base: {color_result['mean_base_delta']:.2f}\n")
            f_log.write(f"dE impresso: {color_result['mean_print_delta']:.2f}\n")
            f_log.write(f"dE impresso colorido: {color_result['mean_color_print_delta']:.2f}\n")
            f_log.write(f"dE p90 colorido: {color_result['p90_color_print_delta']:.2f}\n")
            f_log.write(f"dE comp. local colorido: {color_result['local_color_component_score']:.2f}\n")
            f_log.write(f"dE p95 comp. local: {color_result['local_color_component_p95']:.2f}\n")
            f_log.write(f"Area comp. local: {color_result['local_color_component_area']}\n")
            f_log.write(f"dE efetivo: {color_result['effective_delta']:.2f}\n\n")
            
            print(f"     Cor alterada: {'SIM' if color_result['changed'] else 'NAO'}")
            print(f"     dE efetivo: {color_result['effective_delta']:.2f}")

    f_log.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
