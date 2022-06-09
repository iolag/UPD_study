# a dapted from RD repo
import numpy as np
import cv2
import torch


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap(input: torch.Tensor, anomaly_map):
    # Convert to numpy
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    ano_map = min_max_norm(anomaly_map)
    ano_map = cvt2heatmap(ano_map * 255)
    img = input.permute(1, 2, 0).cpu().numpy() * 255
    img = np.uint8(min_max_norm(img) * 255)
    ano_map = show_cam_on_image(img, ano_map)
    return ano_map
