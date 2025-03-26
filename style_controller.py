import os
import time
import uuid
import json
import logging
import numpy as np
import torch
import dnnlib
import legacy
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import threading
import queue

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StyleGAN2Controller:
    """
    StyleGAN2 모델을 활용한 이미지 스타일 변환 컨트롤러
   
    이 클래스는 StyleGAN2-ADA-PyTorch 스크립트를 래핑하여
    이미지 투영, 스타일 믹싱, Circular Loop 생성 기능을 제공합니다.
    """
   
    def __init__(self, config: Dict):
        """
        StyleGAN2Controller 초기화
       
        Args:
            config: 설정 정보를 담은 딕셔너리
                - stylegan_dir: StyleGAN2-ADA-PyTorch 디렉토리 경로
                - model_path: StyleGAN2 모델 파일 경로 (.pkl)
                - output_dir: 결과 저장 디렉토리
                - cache_dir: 캐시 저장 디렉토리
        """
        self.stylegan_dir = Path(config.get('stylegan_dir', '/home/elicer/stylegan2-ada-pytorch'))
        self.model_path = Path(config.get('model_path', '/home/elicer/stylegan2-ada-pytorch/results/00000-top_cat_256-mirror-24gb-gpu-gamma50-kimg2000-bg-resumecustom/network-snapshot-002000.pkl'))
        self.output_dir = Path(config.get('output_dir', '/home/elicer/stylegan2-ada-pytorch/content/output'))
        self.cache_dir = Path(config.get('cache_dir', './cache'))
       
        # 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
       
        # StyleGAN2 스크립트 경로
        self.pbaylies_projector_script = self.stylegan_dir / "pbaylies_projector.py"
        self.generate_script = self.stylegan_dir / "generate.py"
        self.style_mixing_script = self.stylegan_dir / "style_mixing.py"
       
        # 스크립트 존재 여부 확인
        self._validate_scripts()
       
        # 모델 정보 캐싱
        self.model_info = {
            'path': str(self.model_path),
            'name': self.model_path.stem,
        }
       
        # 작업 관리
        self.jobs = {}
        
        logger.info(f"StyleGAN2Controller 초기화 완료: 모델={self.model_path.name}")
   
    def _validate_scripts(self):
        """필요한 StyleGAN2 스크립트 존재 여부 확인"""
        required_scripts = [
            self.pbaylies_projector_script,
            self.generate_script,
            self.style_mixing_script
        ]
       
        for script in required_scripts:
            if not script.exists():
                raise FileNotFoundError(f"필요한 스크립트를 찾을 수 없습니다: {script}")
    
    def preprocess_image(self, image_path: Union[str, Path]) -> Path:
        """
        이미지 전처리: 크기 조정 및 포맷 변환
       
        Args:
            image_path: 입력 이미지 경로
           
        Returns:
            전처리된 이미지 경로
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
       
        # 이미지 열기
        img = Image.open(image_path)
       
        # RGBA인 경우 RGB로 변환
        if img.mode == 'RGBA':
            img = img.convert('RGB')
       
        # 타겟 크기
        target_size = (512, 512)
       
        # 비율 유지하며 리사이징
        img.thumbnail(target_size, Image.LANCZOS)
       
        # 정사각형 이미지로 만들기
        new_img = Image.new("RGB", target_size, (255, 255, 255))
        paste_x = (target_size[0] - img.width) // 2
        paste_y = (target_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
       
        # 전처리된 이미지 저장
        processed_dir = self.cache_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
       
        processed_path = processed_dir / f"processed_{uuid.uuid4().hex}.png"
        new_img.save(processed_path)
       
        logger.info(f"이미지 전처리 완료: {processed_path}")
        return processed_path
   
    def project_image(self,
                     image_path: Union[str, Path],
                     num_steps: int = 3000,
                     use_vgg: bool = False,
                     use_clip: bool = False,
                     use_center: bool = False,
                     seed: int = 42) -> Dict:
        """
        이미지를 StyleGAN2 잠재 공간에 투영 (pbaylies_projector만 사용)
       
        Args:
            image_path: 입력 이미지 경로
            num_steps: 투영 단계 수
            use_vgg: VGG 특성 기반 손실 사용 여부
            use_clip: CLIP 기반 손실 사용 여부
            use_center: 중앙 이미지 크롭 최적화 사용 여부
            seed: 랜덤 시드
           
        Returns:
            투영 결과 정보 딕셔너리
        """
        # 이미지 전처리
        #processed_path = self.preprocess_image(image_path)
        processed_path = "/home/elicer/stylegan2-ada-pytorch/content/sd-img/stripe1.jpg"

        # 투영 결과 저장 디렉토리
        projection_id = f"proj_{uuid.uuid4().hex}"
        projection_dir = self.output_dir / projection_id
        projection_dir.mkdir(exist_ok=True)
       
        # pbaylies_projector 명령 구성
        cmd = [
            "python", str(self.pbaylies_projector_script),
            f"--network={self.model_path}",
            f"--target-image={processed_path}",
            f"--outdir={projection_dir}",
            f"--num-steps={num_steps}",
            "--use-vgg=False",
            "--use-clip=False",
            f"--seed={seed}"
        ]
       
        logger.info(f"이미지 투영 시작: {cmd}")
       
        # 투영 실행
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
           
            # 결과 파일 확인
            projected_w_path = projection_dir / "projected_w.npz"
            if not projected_w_path.exists():
                raise FileNotFoundError(f"투영 결과 파일을 찾을 수 없습니다: {projected_w_path}")
           
            # 결과 이미지 경로 찾기
            result_images = list(projection_dir.glob("step*.png"))
            final_image = None
            if result_images:
                # step 번호로 정렬하여 마지막 이미지 가져오기
                result_images.sort(key=lambda x: int(x.stem.replace("step", "")))
                final_image = result_images[-1]
            
            # 원본 이미지에 가장 가까운 이미지 찾기
            proj_img_path = projection_dir / "proj.png"
            if proj_img_path.exists():
                final_image = proj_img_path
           
            # 메타데이터 저장
            metadata = {
                "projection_id": projection_id,
                "original_image": str(image_path),
                "processed_image": str(processed_path),
                "projected_w_path": str(projected_w_path),
                "final_image": str(final_image) if final_image else None,
                "num_steps": num_steps,
                "timestamp": time.time()
            }
           
            with open(projection_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
           
            logger.info(f"이미지 투영 완료: {projection_id}")
            return metadata
           
        except subprocess.CalledProcessError as e:
            logger.error(f"이미지 투영 실패: {e.stderr}")
            raise RuntimeError(f"이미지 투영 실패: {e.stderr}")
       
        except Exception as e:
            logger.error(f"이미지 투영 중 오류 발생: {e}")
            raise
    
    def create_mixed_w_vector(self, 
                          source_w_path: Union[str, Path], 
                          target_w_path: Union[str, Path] = None, 
                          target_seed: int = None,
                          layer_cutoff: Union[int, str] = 8,
                          truncation_psi: float = 0.7,
                          output_path: Union[str, Path] = None) -> Path:
        """
        직접 W 벡터 믹싱을 수행하고 혼합된 W 벡터를 NPZ 파일로 저장합니다.
        
        StyleGAN2의 style_mixing.py는 혼합된 W 벡터를 저장하지 않기 때문에,
        이 함수는 비슷한 로직을 구현하여 혼합된 W 벡터를 직접 생성하고 저장합니다.
        
        Args:
            source_w_path: 소스 W 벡터 경로 (투영된 이미지나 시드에서 생성)
            target_w_path: 타겟 W 벡터 경로 (선택적, target_seed와 함께 사용 불가)
            target_seed: 타겟 시드 (선택적, target_w_path와 함께 사용 불가)
            layer_cutoff: 레이어 커트오프 지점 또는 범위:
                        - 정수인 경우 (예: 8): 0-7 레이어는 소스, 8-끝 레이어는 타겟
                        - 문자열 범위인 경우 (예: '0-6'): 지정된 범위의 레이어만 타겟으로 설정
            truncation_psi: 트런케이션 팩터 (기본값: 0.7)
            output_path: 출력 파일 경로 (기본값: 자동 생성)
            
        Returns:
            혼합된 W 벡터의 NPZ 파일 경로
        """
        import torch
        import numpy as np
        import dnnlib
        import legacy
        import re
        from pathlib import Path
        
        # 경로 확인
        source_w_path = Path(source_w_path)
        if not source_w_path.exists():
            raise FileNotFoundError(f"소스 W 벡터 파일을 찾을 수 없습니다: {source_w_path}")
        
        # 출력 경로 설정
        if output_path is None:
            output_dir = self.cache_dir / "mixed_w"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"mixed_w_{uuid.uuid4().hex[:8]}.npz"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # style_mixing.py의 num_range 함수와 유사한 범위 파싱 함수
        def parse_layer_range(layer_spec, num_layers):
            """
            레이어 범위 문자열을 파싱하여 레이어 인덱스 리스트 반환
            예: '0-6' -> [0, 1, 2, 3, 4, 5, 6], '8' -> [8], '0,4,8' -> [0, 4, 8]
            """
            if isinstance(layer_spec, int):
                # 정수인 경우 해당 레이어부터 끝까지
                return list(range(layer_spec, num_layers))
            
            if isinstance(layer_spec, str):
                # style_mixing.py의 num_range 함수와 유사하게 처리
                range_re = re.compile(r'^(\d+)-(\d+)$')
                m = range_re.match(layer_spec)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2))
                    if end >= num_layers:
                        end = num_layers - 1
                    return list(range(start, end+1))
                
                # 쉼표로 구분된 숫자 목록
                if ',' in layer_spec:
                    return [int(x) for x in layer_spec.split(',')]
                
                # 단일 숫자
                if layer_spec.isdigit():
                    layer = int(layer_spec)
                    if layer < num_layers:
                        return [layer]
                    else:
                        return []
            
            # 기본적으로 빈 리스트 반환
            return []
        
        try:
            # 모델 로드 - Path 객체를 str로 변환하여 전달
            logger.info(f"StyleGAN2 모델 로드 중: {self.model_path}")
            with dnnlib.util.open_url(str(self.model_path)) as f:  # Convert Path to string
                G = legacy.load_network_pkl(f)['G_ema'].to(device)
            
            # 소스 W 벡터 로드 - Path 객체를 str로 변환하여 사용
            logger.info(f"소스 W 벡터 로드 중: {source_w_path}")
            source_w_data = np.load(str(source_w_path))  # Convert Path to string
            source_w = torch.from_numpy(source_w_data['w']).to(device)
            
            # 타겟 W 벡터 준비
            if target_w_path is not None and target_seed is not None:
                raise ValueError("target_w_path와 target_seed는 동시에 지정할 수 없습니다.")
            
            if target_w_path is not None:
                # 파일에서 타겟 W 벡터 로드 - Path 객체를 str로 변환하여 사용
                target_w_path = Path(target_w_path)
                if not target_w_path.exists():
                    raise FileNotFoundError(f"타겟 W 벡터 파일을 찾을 수 없습니다: {target_w_path}")
                
                logger.info(f"타겟 W 벡터 로드 중: {target_w_path}")
                target_w_data = np.load(str(target_w_path))  # Convert Path to string
                target_w = torch.from_numpy(target_w_data['w']).to(device)
            
            elif target_seed is not None:
                # 시드에서 타겟 W 벡터 생성
                logger.info(f"시드 {target_seed}에서 타겟 W 벡터 생성 중")
                z = torch.from_numpy(np.random.RandomState(target_seed).randn(1, G.z_dim)).to(device)
                target_w = G.mapping(z, None)
                
                # 트런케이션 적용
                w_avg = G.mapping.w_avg
                target_w = w_avg + (target_w - w_avg) * truncation_psi
            
            else:
                raise ValueError("타겟 W 벡터(target_w_path) 또는 타겟 시드(target_seed)를 지정해야 합니다.")
            
            # W 벡터 형태 확인 및 일치
            # StyleGAN2의 W 벡터는 일반적으로 [batch_size, num_layers, latent_dim] 또는 
            # [batch_size, latent_dim] 형태일 수 있음
            
            # 소스 W 벡터가 [latent_dim] 또는 [1, latent_dim] 형태인 경우 확장
            if source_w.ndim == 1:
                source_w = source_w.unsqueeze(0)  # [latent_dim] -> [1, latent_dim]
            
            if source_w.ndim == 2 and G.num_ws > 1:
                # 단일 w 값을 모든 레이어로 복제 [1, latent_dim] -> [1, num_layers, latent_dim]
                if source_w.shape[0] == 1:
                    source_w = source_w.unsqueeze(1).repeat(1, G.num_ws, 1)
                else:  # [batch, latent_dim] -> [batch, num_layers, latent_dim]
                    source_w = source_w.unsqueeze(1).repeat(1, G.num_ws, 1)
            
            # 타겟 W 벡터에 대해서도 동일한 과정 수행
            if target_w.ndim == 1:
                target_w = target_w.unsqueeze(0)
            
            if target_w.ndim == 2 and G.num_ws > 1:
                if target_w.shape[0] == 1:
                    target_w = target_w.unsqueeze(1).repeat(1, G.num_ws, 1)
                else:
                    target_w = target_w.unsqueeze(1).repeat(1, G.num_ws, 1)
            
            # 레이어 범위 파싱
            logger.info(f"레이어 커트오프 {layer_cutoff}에서 W 벡터 믹싱 중")
            mixed_w = source_w.clone()
            
            # 타겟 스타일을 적용할 레이어 인덱스 결정
            target_layers = parse_layer_range(layer_cutoff, G.num_ws)
            
            if not target_layers:
                raise ValueError(f"유효한 레이어 범위가 지정되지 않았습니다: {layer_cutoff}")
            
            # 지정된 레이어에 타겟 스타일 적용
            for layer_idx in target_layers:
                mixed_w[0, layer_idx] = target_w[0, layer_idx]
            
            # 혼합된 W 벡터 저장 - Path 객체를 str로 변환하여 사용
            logger.info(f"혼합된 W 벡터 저장 중: {output_path}")
            np.savez(str(output_path), w=mixed_w.cpu().numpy())  # Convert Path to string
            
            return output_path
        
        except Exception as e:
            logger.error(f"W 벡터 믹싱 중 오류 발생: {e}")
            raise RuntimeError(f"W 벡터 믹싱 실패: {str(e)}")

    def generate_from_mixed_w(self,
                             mixed_w_path: Union[str, Path],
                             output_dir: Union[str, Path] = None,
                             truncation_psi: float = 0.7) -> Dict:
        """
        혼합된 W 벡터로부터 이미지 생성
        
        Args:
            mixed_w_path: 혼합된 W 벡터 NPZ 파일 경로
            output_dir: 출력 이미지 저장 디렉토리 (기본값: 자동 생성)
            truncation_psi: 트런케이션 팩터 (기본값: 0.7)
            
        Returns:
            생성 결과 정보 딕셔너리
        """
        # 경로 확인
        mixed_w_path = Path(mixed_w_path)
        if not mixed_w_path.exists():
            raise FileNotFoundError(f"혼합된 W 벡터 파일을 찾을 수 없습니다: {mixed_w_path}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            result_id = f"mixed_result_{uuid.uuid4().hex[:8]}"
            output_dir = self.output_dir / result_id
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # generate.py 실행 - 모든 Path 객체를 str로 변환
        cmd = [
            "python", str(self.generate_script),
            f"--network={str(self.model_path)}",
            f"--projected-w={str(mixed_w_path)}",
            f"--outdir={str(output_dir)}",
            f"--trunc={truncation_psi}"
        ]
        
        logger.info(f"혼합된 W 벡터로부터 이미지 생성 중: {cmd}")
        
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # 결과 이미지 확인
            result_images = list(output_dir.glob("*.png"))
            result_image = None
            
            if result_images:
                result_image = result_images[0]
            
            # 메타데이터 저장
            metadata = {
                "mixed_w_path": str(mixed_w_path),
                "output_dir": str(output_dir),
                "result_image": str(result_image) if result_image else None,
                "truncation_psi": truncation_psi,
                "timestamp": time.time()
            }
            
            # Path를 str로 변환하여 메타데이터 저장
            metadata_path = output_dir / "metadata.json"
            with open(str(metadata_path), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"이미지 생성 완료: {output_dir}")
            return metadata
            
        except subprocess.CalledProcessError as e:
            logger.error(f"이미지 생성 실패: {e.stderr}")
            raise RuntimeError(f"이미지 생성 실패: {e.stderr}")
            
        except Exception as e:
            logger.error(f"이미지 생성 중 오류 발생: {e}")
            raise

    def custom_style_mixing(self,
                      source_w_path: Union[str, Path],
                      target_w_path: Union[str, Path],
                      style_range: Union[str, int] = "0-6",
                      truncation_psi: float = 0.7,
                      output_dir: Union[str, Path] = None,
                      generate_grid: bool = True) -> Dict:
        """
        프로젝션된 W 벡터 간의 스타일 믹싱을 수행하는 사용자 정의 함수
        
        기존 style_mixing.py 스크립트는 랜덤 시드에서 W 벡터를 생성하도록 설계되었으나,
        이 함수는 이미 생성된 W 벡터(projected_w.npz)를 직접 사용합니다.
        
        Args:
            source_w_path: 소스 W 벡터 경로 (구조적 특성)
            target_w_path: 타겟 W 벡터 경로 (스타일 특성)
            style_range: 스타일 믹싱을 적용할 레이어 범위 (예: "0-6", 8, "0,4,8")
            truncation_psi: 트런케이션 팩터
            output_dir: 출력 디렉토리
            generate_grid: 그리드 이미지 생성 여부
            
        Returns:
            스타일 믹싱 결과 정보 딕셔너리
        """
        import torch
        import numpy as np
        import dnnlib
        import legacy
        from pathlib import Path
        import PIL.Image
        import re
        
        # 경로 확인
        source_w_path = Path(source_w_path)
        target_w_path = Path(target_w_path)
        
        if not source_w_path.exists():
            raise FileNotFoundError(f"소스 W 벡터 파일을 찾을 수 없습니다: {source_w_path}")
        
        if not target_w_path.exists():
            raise FileNotFoundError(f"타겟 W 벡터 파일을 찾을 수 없습니다: {target_w_path}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            mixing_id = f"mixing_{uuid.uuid4().hex[:8]}"
            output_dir = self.output_dir / mixing_id
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 스타일 범위 파싱 함수
        def parse_style_range(style_spec, num_layers):
            """레이어 범위 문자열을 파싱하여 레이어 인덱스 리스트 반환"""
            if isinstance(style_spec, int):
                # 정수인 경우 해당 레이어부터 끝까지
                return list(range(style_spec, num_layers))
            
            if isinstance(style_spec, str):
                # 범위 문자열 파싱 (예: "0-6")
                range_re = re.compile(r'^(\d+)-(\d+)$')
                m = range_re.match(style_spec)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2))
                    if end >= num_layers:
                        end = num_layers - 1
                    return list(range(start, end+1))
                
                # 쉼표로 구분된 숫자 목록 (예: "0,4,8")
                if ',' in style_spec:
                    return [int(x) for x in style_spec.split(',')]
                
                # 단일 숫자 (예: "8")
                if style_spec.isdigit():
                    layer = int(style_spec)
                    if layer < num_layers:
                        return list(range(layer, num_layers))
                    else:
                        return []
            
            # 기본적으로 빈 리스트 반환
            return []
        
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # 모델 로드
            logger.info(f"StyleGAN2 모델 로드 중: {self.model_path}")
            with dnnlib.util.open_url(str(self.model_path)) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)
            
            # W 벡터 로드
            logger.info(f"소스 W 벡터 로드 중: {source_w_path}")
            source_w_data = np.load(str(source_w_path))
            source_w = torch.from_numpy(source_w_data['w']).to(device)
            
            logger.info(f"타겟 W 벡터 로드 중: {target_w_path}")
            target_w_data = np.load(str(target_w_path))
            target_w = torch.from_numpy(target_w_data['w']).to(device)
            
            # W 벡터 형태 확인 및 일치
            if source_w.ndim == 1:
                source_w = source_w.unsqueeze(0)
            
            if target_w.ndim == 1:
                target_w = target_w.unsqueeze(0)
            
            # 모든 W 벡터를 확장하여 [batch, num_layers, latent_dim] 형태로 변환
            if source_w.ndim == 2 and G.num_ws > 1:
                if source_w.shape[0] == 1:
                    source_w = source_w.unsqueeze(1).repeat(1, G.num_ws, 1)
                else:
                    source_w = source_w.unsqueeze(1).repeat(1, G.num_ws, 1)
            
            if target_w.ndim == 2 and G.num_ws > 1:
                if target_w.shape[0] == 1:
                    target_w = target_w.unsqueeze(1).repeat(1, G.num_ws, 1)
                else:
                    target_w = target_w.unsqueeze(1).repeat(1, G.num_ws, 1)
            
            # 스타일 범위 파싱
            style_layers = parse_style_range(style_range, G.num_ws)
            if not style_layers:
                raise ValueError(f"유효한 스타일 레이어 범위가 지정되지 않았습니다: {style_range}")
            
            logger.info(f"스타일 믹싱 수행 중: 레이어 {style_layers}")
            
            # 스타일 혼합 W 벡터 생성
            mixed_w = source_w.clone()
            for layer_idx in style_layers:
                mixed_w[0, layer_idx] = target_w[0, layer_idx]
            
            # 이미지 생성
            logger.info("이미지 생성 중...")
            # 소스 이미지
            source_img = G.synthesis(source_w, noise_mode='const')
            source_img = (source_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            
            # 타겟 이미지
            target_img = G.synthesis(target_w, noise_mode='const')
            target_img = (target_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            
            # 혼합 이미지
            mixed_img = G.synthesis(mixed_w, noise_mode='const')
            mixed_img = (mixed_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
            
            # 이미지 저장
            source_path = output_dir / "source.png"
            target_path = output_dir / "target.png"
            mixed_path = output_dir / "mixed.png"
            
            PIL.Image.fromarray(source_img, 'RGB').save(str(source_path))
            PIL.Image.fromarray(target_img, 'RGB').save(str(target_path))
            PIL.Image.fromarray(mixed_img, 'RGB').save(str(mixed_path))
            
            # 혼합된 W 벡터 저장
            mixed_w_path = output_dir / "mixed_w.npz"
            np.savez(str(mixed_w_path), w=mixed_w.cpu().numpy())
            
            # 그리드 이미지 생성 (선택적)
            grid_path = None
            if generate_grid:
                grid_path = output_dir / "grid.png"
                
                # 그리드 이미지 크기 계산
                W = G.img_resolution
                H = G.img_resolution
                
                # 3x1 그리드 (소스, 혼합, 타겟)
                canvas = PIL.Image.new('RGB', (W * 3, H), 'black')
                canvas.paste(PIL.Image.fromarray(source_img, 'RGB'), (0, 0))
                canvas.paste(PIL.Image.fromarray(mixed_img, 'RGB'), (W, 0))
                canvas.paste(PIL.Image.fromarray(target_img, 'RGB'), (W * 2, 0))
                canvas.save(str(grid_path))
            
            # 결과 정보
            result = {
                'source_w_path': str(source_w_path),
                'target_w_path': str(target_w_path),
                'mixed_w_path': str(mixed_w_path),
                'style_range': str(style_range) if isinstance(style_range, int) else style_range,
                'style_layers': style_layers,
                'source_image': str(source_path),
                'target_image': str(target_path),
                'mixed_image': str(mixed_path),
                'grid_image': str(grid_path) if grid_path else None,
                'truncation_psi': truncation_psi,
                'timestamp': time.time()
            }
            
            # 메타데이터 저장
            metadata_path = output_dir / "metadata.json"
            with open(str(metadata_path), "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"스타일 믹싱 완료: {output_dir}")
            return result
        
        except Exception as e:
            logger.error(f"스타일 믹싱 중 오류 발생: {e}")
            raise RuntimeError(f"스타일 믹싱 실패: {str(e)}")    
    
    def style_mix_and_circular_loop(self,
                                source_image_path: Union[str, Path],
                                target_image_path: Optional[Union[str, Path]] = None,
                                layer_cutoff: Union[int, str] = 8,
                                target_seeds: Optional[List[int]] = None,
                                truncation: float = 1.0,
                                diameter: float = 100.0,
                                frames: int = 120,
                                random_seed: int = 42,
                                projection_steps: int = 300) -> Dict:
        """
        이미지 프로젝션 후 스타일 믹싱 및 Circular Loop 생성을 한번에 처리
        
        Args:
            source_image_path: 소스 이미지 경로 (구조적 특성)
            target_image_path: 타겟 이미지 경로 (스타일 특성) - 옵션
            layer_cutoff: 레이어 커트오프 지점 또는 범위:
                        - 정수인 경우 (예: 8): 0-7 레이어는 소스, 8-끝 레이어는 타겟
                        - 문자열 범위인 경우 (예: '0-6'): 지정된 범위의 레이어만 타겟으로 설정
            target_seeds: 타겟 이미지가 없을 경우 사용할 시드 목록
            truncation: 트런케이션 팩터 (0.5-1.0)
            diameter: 원형 루프의 직경 (50-500 권장)
            frames: 생성할 프레임 수
            random_seed: Circular Loop의 랜덤 시드
            projection_steps: 프로젝션 단계 수
        
        Returns:
            결과 정보 딕셔너리
        """
        logger.info("스타일 믹싱 + Circular Loop 과정 시작")
        
        # 작업 ID 생성
        job_id = f"mix_loop_{uuid.uuid4().hex}"
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 소스 이미지 프로젝션 (구조 소스)
            logger.info(f"소스 이미지 프로젝션 시작: {source_image_path}")
            source_proj = self.project_image(
                image_path=source_image_path,
                num_steps=projection_steps,
                use_vgg=False,
                use_clip=False,
                seed=random_seed
            )
            source_w_path = source_proj["projected_w_path"]
            
            # 결과 수집을 위한 변수들
            mixing_results = []
            loop_results = []
            all_frame_images = []
            all_video_paths = []
            
            # 2. 스타일 소스 준비 - 이미지 또는 시드 기반
            style_sources = []
            
            if target_image_path:
                # 타겟 이미지가 있는 경우, 이 이미지를 프로젝션
                logger.info(f"타겟 이미지 프로젝션 시작: {target_image_path}")
                target_proj = self.project_image(
                    image_path=target_image_path,
                    num_steps=projection_steps,
                    use_vgg=False,
                    use_clip=False,
                    seed=random_seed + 1
                )
                style_sources.append({
                    'type': 'projected',
                    'w_path': target_proj["projected_w_path"],
                    'image_path': target_proj["final_image"]
                })
            
            # 시드 기반 스타일 소스
            if target_seeds:
                for i, seed in enumerate(target_seeds):
                    style_sources.append({
                        'type': 'seed',
                        'seed': seed,
                        'name': f"seed_{seed}"
                    })
            elif not target_image_path:
                # 타겟 이미지도 없고 시드도 없으면 기본 시드 사용
                default_seeds = [100, 200, 300, 400, 500]
                for i, seed in enumerate(default_seeds):
                    style_sources.append({
                        'type': 'seed',
                        'seed': seed,
                        'name': f"seed_{seed}"
                    })
            
            # 3. 각 스타일 소스에 대해 스타일 믹싱 및 Circular Loop 생성
            for i, style_source in enumerate(style_sources):
                style_id = f"style_{i}"
                style_dir = job_dir / style_id
                style_dir.mkdir(exist_ok=True)
                
                # 스타일 믹싱 생성
                if style_source['type'] == 'seed':
                    # 시드 기반 스타일 소스
                    seed = style_source['seed']
                    style_name = style_source['name']
                    
                    # 시드에서 혼합된 W 벡터 직접 생성
                    mixed_w_path = self.create_mixed_w_vector(
                        source_w_path=source_w_path,
                        target_seed=seed,
                        layer_cutoff=layer_cutoff,  # 이제 int 또는 str 모두 처리 가능
                        truncation_psi=truncation,
                        output_path=style_dir / "mixed_w.npz"
                    )
                    
                    # 시드 기반의 경우 기존 style_mixing.py 스크립트 사용 가능
                    # style_mixing.py 실행 - 모든 Path를 str로 변환
                    mixing_cmd = [
                        "python", str(self.style_mixing_script),
                        f"--network={str(self.model_path)}",
                        f"--rows={random_seed}",  # 소스 시드 - 임의로 사용
                        f"--cols={seed}",  # 타겟 시드
                        f"--styles={layer_cutoff if isinstance(layer_cutoff, str) else f'{layer_cutoff}-'}",
                        f"--trunc={truncation}",
                        f"--outdir={str(style_dir)}"
                    ]
                    
                    logger.info(f"스타일 믹싱 실행 (시드 기반): {mixing_cmd}")
                    
                    try:
                        # 스타일 믹싱 실행 - 오류 발생 시 무시하고 계속 진행
                        subprocess.run(
                            mixing_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True
                        )
                    except subprocess.CalledProcessError:
                        logger.warning(f"기존 style_mixing.py 스크립트 실행 실패 - 계속 진행합니다.")
                    
                    # 위에서 생성한 W 벡터 기반으로 이미지 생성
                    mixing_result = self.generate_from_mixed_w(
                        mixed_w_path=mixed_w_path,
                        output_dir=style_dir / "mixed_result",
                        truncation_psi=truncation
                    )
                    mixing_result_path = mixing_result.get('result_image')
                    
                else:
                    # 프로젝션된 이미지 기반 스타일 소스
                    target_w_path = style_source['w_path']
                    
                    # 사용자 정의 스타일 믹싱 함수 사용
                    logger.info(f"사용자 정의 스타일 믹싱 실행: 소스={source_w_path}, 타겟={target_w_path}")
                    mixing_result = self.custom_style_mixing(
                        source_w_path=source_w_path,
                        target_w_path=target_w_path,
                        style_range=layer_cutoff,
                        truncation_psi=truncation,
                        output_dir=style_dir,
                        generate_grid=True
                    )
                    
                    # 혼합된 W 벡터 경로 및 결과 이미지 경로 추출
                    mixed_w_path = mixing_result['mixed_w_path']
                    mixing_result_path = mixing_result.get('grid_image') or mixing_result.get('mixed_image')
                
                # 4. Circular Loop 생성
                loop_dir = style_dir / "loop"
                loop_dir.mkdir(exist_ok=True)
                
                # generate.py로 circular loop 생성 - 모든 Path를 str로 변환
                loop_cmd = [
                    "python", str(self.generate_script),
                    f"--outdir={str(loop_dir)}",
                    f"--trunc={truncation}",
                    "--process=interpolation",
                    "--interpolation=circularloop",
                    f"--diameter={diameter}",
                    f"--frames={frames}",
                    f"--random_seed={random_seed + i}",  # 각 스타일마다 다른 시드
                    f"--network={str(self.model_path)}",
                    f"--projected-w={str(mixed_w_path)}"  # 이제 혼합된 W 벡터 사용
                ]
                
                logger.info(f"Circular Loop 생성: {loop_cmd}")
                
                # Circular Loop 실행
                subprocess.run(
                    loop_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # 생성된 프레임 이미지 수집
                frame_images = sorted(list(loop_dir.glob("*.png")))
                all_frame_images.extend(frame_images)
                
                # 비디오 생성
                video_path = style_dir / f"style_{i}_loop.mp4"
                
                if frame_images:
                    # FFmpeg 명령어 생성 - 경로를 str로 변환
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",  # 기존 파일 덮어쓰기
                        "-framerate", "24",  # 초당 프레임 수
                        "-pattern_type", "glob",
                        "-i", f"{str(loop_dir)}/*.png",  # 경로를 str로 변환
                        "-c:v", "libx264",
                        "-profile:v", "high",
                        "-crf", "20",
                        "-pix_fmt", "yuv420p",
                        str(video_path)  # 경로를 str로 변환
                    ]
                    
                    logger.info(f"비디오 생성: {ffmpeg_cmd}")
                    
                    # FFmpeg 실행
                    subprocess.run(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    all_video_paths.append(video_path)
                
                # 결과 정보 저장
                mix_result = {
                    'style_id': style_id,
                    'style_source': style_source,
                    'mixing_result': mixing_result_path,
                    'mixed_w_path': str(mixed_w_path),  # 이제 혼합된 W 벡터 경로 포함
                    'frame_images': [str(img) for img in frame_images],
                    'video_path': str(video_path) if video_path.exists() else None
                }
                
                mixing_results.append(mix_result)
                loop_results.append({
                    'style_id': style_id,
                    'frames': len(frame_images),
                    'video_path': str(video_path) if video_path.exists() else None
                })
            
            # 모든 스타일에 대한 통합 결과 비디오 생성 (선택적)
            if all_frame_images:
                # 모든 프레임을 하나의 디렉토리로 복사하고 번호 매기기
                combined_frames_dir = job_dir / "combined_frames"
                combined_frames_dir.mkdir(exist_ok=True)
                
                for i, frame_path in enumerate(all_frame_images):
                    # 파일 복사 - Path 객체를 str로 변환
                    shutil.copy(str(frame_path), str(combined_frames_dir / f"frame_{i:04d}.png"))
                
                # 통합 비디오 생성
                combined_video_path = job_dir / "combined_style_mixing_loop.mp4"
                
                # FFmpeg 명령어에 Path 객체를 str로 변환
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate", "24",
                    "-pattern_type", "glob",
                    "-i", f"{str(combined_frames_dir)}/frame_*.png",
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    str(combined_video_path)
                ]
                
                logger.info(f"통합 비디오 생성: {ffmpeg_cmd}")
                
                # FFmpeg 실행
                subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                all_video_paths.append(combined_video_path)
            
            # 최종 결과 메타데이터 준비
            result = {
                "job_id": job_id,
                "source_image": str(source_image_path),
                "target_image": str(target_image_path) if target_image_path else None,
                "source_projected_w": str(source_w_path),
                "layer_cutoff": str(layer_cutoff) if isinstance(layer_cutoff, int) else layer_cutoff,
                "truncation": truncation,
                "diameter": diameter,
                "frames": frames,
                "random_seed": random_seed,
                "mixing_results": mixing_results,
                "loop_results": loop_results,
                "all_video_paths": [str(video) for video in all_video_paths],
                "all_frame_count": len(all_frame_images),
                "timestamp": time.time()
            }
            
            # 메타데이터 저장 - Path 객체를 str로 변환
            metadata_path = job_dir / "metadata.json"
            with open(str(metadata_path), "w") as f:
                json.dump(result, f, indent=2)
            
            # 작업 정보 저장
            self.jobs[job_id] = {
                "status": "completed",
                "result": result,
                "message": "Style Mixing + Circular Loop 생성 완료"
            }
            
            logger.info(f"Style Mixing + Circular Loop 생성 완료: {job_id}")
            return result
        
        except Exception as e:
            error_message = f"Style Mixing + Circular Loop 생성 오류: {str(e)}"
            logger.error(error_message)
            
            # 오류 정보 저장
            self.jobs[job_id] = {
                "status": "failed",
                "error": str(e),
                "message": error_message
            }
            
            # 오류 메타데이터 저장 - Path 객체를 str로 변환
            error_path = job_dir / "error.json"
            error_data = {
                "job_id": job_id,
                "error": str(e),
                "source_image": str(source_image_path),
                "target_image": str(target_image_path) if target_image_path else None,
                "timestamp": time.time()
            }
            
            with open(str(error_path), "w") as f:
                json.dump(error_data, f, indent=2)
            
            raise RuntimeError(error_message)
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        작업 상태 확인
        
        Args:
            job_id: 작업 ID
            
        Returns:
            작업 상태 정보
        """
        if job_id not in self.jobs:
            # 작업 정보가 메모리에 없으면 파일에서 확인
            job_dir = self.output_dir / job_id
            
            if not job_dir.exists():
                return {
                    "status": "not_found",
                    "message": f"작업 ID {job_id}를 찾을 수 없습니다."
                }
            
            # 메타데이터 또는 오류 파일 확인
            metadata_path = job_dir / "metadata.json"
            error_path = job_dir / "error.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    result = json.load(f)
                
                self.jobs[job_id] = {
                    "status": "completed",
                    "result": result,
                    "message": "작업 완료"
                }
            
            elif error_path.exists():
                with open(error_path, "r") as f:
                    error_data = json.load(f)
                
                self.jobs[job_id] = {
                    "status": "failed",
                    "error": error_data.get("error", "알 수 없는 오류"),
                    "message": "작업 실패"
                }
            
            else:
                # 작업 디렉토리는 있지만 메타데이터나 오류 파일이 없는 경우
                self.jobs[job_id] = {
                    "status": "processing",
                    "message": "작업 처리 중"
                }
        
        return self.jobs[job_id]


# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        "stylegan_dir": "/home/elicer/stylegan2-ada-pytorch",
        "model_path": "/home/elicer/stylegan2-ada-pytorch/results/00000-top_cat_256-mirror-24gb-gpu-gamma50-kimg2000-bg-resumecustom/network-snapshot-002000.pkl",
        "output_dir": "/home/elicer/stylegan2-ada-pytorch/content/output",
        "cache_dir": "./cache"
    }
    
    # 컨트롤러 초기화
    controller = StyleGAN2Controller(config)
    
    # 이미지 투영 및 스타일 믹싱 + Circular Loop 생성 예시
    try:
        # 이미지 투영 및 스타일 믹싱 + Circular Loop 생성
        result = controller.style_mix_and_circular_loop(
            source_image_path="source_image.jpg",
            target_image_path="target_image.jpg",
            layer_cutoff='0-6',
            truncation=0.8,
            diameter=200.0,
            frames=120,
            random_seed=42
        )
        
        print("작업 완료! 결과:", result["job_id"])
        
    except Exception as e:
        print(f"오류 발생: {e}")