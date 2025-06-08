import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import exposure
from typing import Tuple, Optional

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ImageEnhancer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.srcnn = SRCNN().to(device)
        # モデルの重みがある場合はロード
        try:
            self.srcnn.load_state_dict(torch.load('models/srcnn.pth'))
        except:
            print("SRCNNモデルの重みが見つかりません。デフォルトの重みを使用します。")
        self.srcnn.eval()

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE（コントラスト制限付き適応的ヒストグラム均一化）を適用
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l,a,b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def adjust_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        画像の明るさを自動調整
        """
        # 画像の平均明るさを計算
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # 目標の明るさ（128）に基づいてガンマ値を計算
        gamma = np.log(128) / np.log(mean_brightness + 1e-6)
        gamma = min(max(gamma, 0.5), 2.0)  # ガンマ値を0.5から2.0の範囲に制限
        
        return exposure.adjust_gamma(image, gamma)

    def apply_super_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        SRCNNを使用して超解像を適用
        """
        # 入力画像の前処理
        img = image.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            output = self.srcnn(img)
            output = output.clamp(0.0, 1.0)

        # 出力画像の後処理
        output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        output = (output * 255.0).astype(np.uint8)
        
        return output

    def enhance_image(self, image: np.ndarray, 
                     apply_sr: bool = True,
                     apply_clahe: bool = True,
                     adjust_brightness: bool = True) -> np.ndarray:
        """
        画像に複数の改善処理を適用
        """
        enhanced = image.copy()
        
        # 明るさの自動調整
        if adjust_brightness:
            enhanced = self.adjust_gamma(enhanced)
        
        # CLAHEの適用
        if apply_clahe:
            enhanced = self.apply_clahe(enhanced)
        
        # 超解像の適用
        if apply_sr:
            enhanced = self.apply_super_resolution(enhanced)
        
        return enhanced

    def analyze_image_quality(self, image: np.ndarray) -> dict:
        """
        画像の品質を分析
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ラプラシアンによるブレ検出
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # コントラストの計算
        contrast = gray.std()
        
        # 明るさの計算
        brightness = np.mean(gray)
        
        return {
            'sharpness': laplacian_var,
            'contrast': contrast,
            'brightness': brightness
        }

    def get_enhancement_params(self, image: np.ndarray) -> Tuple[bool, bool, bool]:
        """
        画像分析に基づいて、どの改善処理を適用するか決定
        """
        quality_metrics = self.analyze_image_quality(image)
        
        # 各処理の適用基準
        apply_sr = quality_metrics['sharpness'] < 100  # ブレが大きい場合
        apply_clahe = quality_metrics['contrast'] < 50  # コントラストが低い場合
        adjust_brightness = not (40 < quality_metrics['brightness'] < 200)  # 明るさが適切でない場合
        
        return apply_sr, apply_clahe, adjust_brightness 