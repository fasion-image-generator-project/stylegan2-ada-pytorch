#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StyleGAN2-ADA 통합 테스트 스크립트

이 스크립트는 Linux 인스턴스에서 StyleGAN2-ADA 모델의 이미지 변환 및
스타일 조정 기능을 테스트합니다.

실행 방법:
    python stylegan_integration_test.py --stylegan_dir=/path/to/stylegan2-ada-pytorch --model=/path/to/network.pkl
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
import logging
import json
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stylegan_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StyleGANTester:
    def __init__(self, args):
        """테스트 환경 초기화"""
        self.stylegan_dir = Path(args.stylegan_dir).resolve()
        self.model_path = Path(args.model).resolve()
        self.output_dir = Path(args.output_dir).resolve()
        self.test_image = args.test_image
        self.seed = args.seed
        
        # 출력 디렉토리 구조 생성
        self.test_id = f"test_{int(time.time())}"
        self.test_dir = self.output_dir / self.test_id
        self.images_dir = self.test_dir / "images"
        self.projections_dir = self.test_dir / "projections"
        self.factors_dir = self.test_dir / "factors"
        self.results_dir = self.test_dir / "results"
        
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.projections_dir, exist_ok=True)
        os.makedirs(self.factors_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 테스트 결과 추적
        self.results = {
            "environment_check": False,
            "test_image_preparation": False,
            "projection_test": False,
            "factor_extraction_test": False,
            "factor_application_test": False,
            "style_mixing_test": False,
            "overall_success": False
        }
        
        logger.info(f"테스트 환경 초기화 완료. 테스트 ID: {self.test_id}")
        logger.info(f"StyleGAN2-ADA 디렉토리: {self.stylegan_dir}")
        logger.info(f"모델 경로: {self.model_path}")
        logger.info(f"출력 디렉토리: {self.test_dir}")
        
    def check_environment(self):
        """환경 확인: GPU 및 필요한 라이브러리"""
        try:
            logger.info("환경 확인 중...")
            
            # GPU 확인
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU 확인 완료: {gpu_count}개 발견, 첫 번째 GPU: {gpu_name}")
            else:
                logger.error("GPU를 찾을 수 없습니다!")
                return False

            # 필요한 파일 확인
            required_files = [
                self.stylegan_dir / "pbaylies_projector.py",
                self.stylegan_dir / "closed_form_factorization.py",
                self.stylegan_dir / "apply_factor.py",
                self.stylegan_dir / "style_mixing.py"
            ]
            
            for file in required_files:
                if not file.exists():
                    logger.error(f"필요한 파일을 찾을 수 없습니다: {file}")
                    return False
            
            # 모델 파일 확인
            if not self.model_path.exists():
                logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
                
            logger.info("환경 확인 완료: 모든 요구 사항이 충족됩니다.")
            self.results["environment_check"] = True
            return True
            
        except Exception as e:
            logger.error(f"환경 확인 중 오류 발생: {e}")
            return False
    
    def prepare_test_image(self):
        """테스트 이미지 준비"""
        try:
            logger.info("테스트 이미지 준비 중...")
            
            if self.test_image:
                # 사용자 지정 이미지 사용
                image_path = Path(self.test_image)
                if not image_path.exists():
                    logger.error(f"지정된 테스트 이미지를 찾을 수 없습니다: {image_path}")
                    return False
                
                # 이미지 크기 확인 및 조정
                img = Image.open(image_path)
                target_size = (512, 512)
                
                # 비율 유지하면서 리사이징
                img.thumbnail(target_size, Image.LANCZOS)
                
                # 정사각형 이미지로 만들기
                new_img = Image.new("RGB", target_size, (255, 255, 255))
                paste_x = (target_size[0] - img.width) // 2
                paste_y = (target_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # 테스트 디렉토리에 저장
                self.test_image_path = self.images_dir / "test_image.png"
                new_img.save(self.test_image_path)
                
            else:
                # 샘플 이미지 다운로드
                url = "https://github.com/NVlabs/stylegan2-ada-pytorch/raw/main/docs/stylegan2-ada-teaser-1024x1024.png"
                sample_image_path = self.images_dir / "sample.png"
                
                logger.info(f"샘플 이미지 다운로드 중: {url}")
                urllib.request.urlretrieve(url, sample_image_path)
                
                # 이미지 크기 조정 (512x512)
                img = Image.open(sample_image_path)
                img = img.resize((512, 512), Image.LANCZOS)
                
                # 테스트 이미지로 저장
                self.test_image_path = self.images_dir / "test_image.png"
                img.save(self.test_image_path)
            
            # 실패 시 랜덤 이미지 생성
            if not os.path.exists(self.test_image_path):
                logger.warning("이미지 준비 실패, 랜덤 이미지 생성 중...")
                
                # 모델을 사용하여 이미지 생성
                cmd = [
                    "python", str(self.stylegan_dir / "generate.py"),
                    f"--network={self.model_path}",
                    f"--seeds={self.seed}",
                    f"--outdir={self.images_dir}"
                ]
                
                subprocess.run(cmd, check=True)
                self.test_image_path = list(self.images_dir.glob(f"seed{self.seed:04d}.png"))[0]
                
                if not os.path.exists(self.test_image_path):
                    logger.error("이미지 생성 실패")
                    return False
            
            logger.info(f"테스트 이미지 준비 완료: {self.test_image_path}")
            self.results["test_image_preparation"] = True
            return True
            
        except Exception as e:
            logger.error(f"테스트 이미지 준비 중 오류 발생: {e}")
            return False
    
    def test_projection(self):
        """이미지 투영 테스트"""
        try:
            logger.info("이미지 투영 테스트 중...")
            
            # pbaylies_projector.py 실행
            cmd = [
                "python", str(self.stylegan_dir / "pbaylies_projector.py"),
                f"--network={self.model_path}",
                f"--target-image={self.test_image_path}",
                f"--outdir={self.projections_dir}",
                "--num-steps=200",  # 테스트용으로 짧게 설정
                "--use-clip=False",
                "--use-center=False",
                "--seed=42"
            ]
            
            logger.info(f"실행 명령: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"투영 실패: {process.stderr}")
                return False
            
            # 프로젝션 결과 파일 확인
            projected_w_path = self.projections_dir / "projected_w.npz"
            if not projected_w_path.exists():
                logger.error("투영 결과 파일을 찾을 수 없습니다")
                return False
            
            # 중간 이미지 확인
            final_image = list(self.projections_dir.glob("proj*.png"))
            if not final_image:
                logger.error("투영 이미지 결과를 찾을 수 없습니다")
                return False
            
            self.projected_w_path = projected_w_path
            logger.info(f"이미지 투영 테스트 완료: {projected_w_path}")
            self.results["projection_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"이미지 투영 테스트 중 오류 발생: {e}")
            return False
    
    def test_closed_form_factorization(self):
        """스타일 요소 추출 테스트"""
        try:
            logger.info("스타일 요소 추출 테스트 중...")
            
            # closed_form_factorization.py 실행
            factors_path = self.factors_dir / "factors.pt"
            cmd = [
                "python", str(self.stylegan_dir / "closed_form_factorization.py"),
                f"--ckpt={self.model_path}",
                f"--out={factors_path}"
            ]
            
            logger.info(f"실행 명령: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"스타일 요소 추출 실패: {process.stderr}")
                return False
            
            # 결과 파일 확인
            if not factors_path.exists():
                logger.error(f"스타일 요소 파일을 찾을 수 없습니다: {factors_path}")
                return False
            
            self.factors_path = factors_path
            logger.info(f"스타일 요소 추출 테스트 완료: {factors_path}")
            self.results["factor_extraction_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"스타일 요소 추출 테스트 중 오류 발생: {e}")
            return False
    
    def test_apply_factor(self):
        """스타일 요소 적용 테스트"""
        try:
            logger.info("스타일 요소 적용 테스트 중...")
            
            if not hasattr(self, 'factors_path') or not hasattr(self, 'projected_w_path'):
                logger.error("필요한 파일 경로가 설정되지 않았습니다")
                return False
            
            # 여러 요소와 강도로 테스트
            factors_to_test = [0, 1, 2, 3, 4]  # 상위 5개 요소 테스트
            strengths = [5.0, 10.0]         # 강도 테스트
            
            for factor_idx in factors_to_test:
                for strength in strengths:
                    factor_output_dir = self.results_dir / f"factor_{factor_idx}_strength_{strength}"
                    os.makedirs(factor_output_dir, exist_ok=True)
                    
                    # apply_factor.py 실행
                    cmd = [
                        "python", str(self.stylegan_dir / "apply_factor.py"),
                        "-i", str(factor_idx),
                        "-d", str(strength),
                        "--ckpt", str(self.model_path),
                        str(self.factors_path),
                        "--output", str(factor_output_dir),
                        "--video"
                    ]
                    
                    logger.info(f"요소 {factor_idx}, 강도 {strength} 적용 중...")
                    process = subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    if process.returncode != 0:
                        logger.warning(f"요소 {factor_idx} 적용 실패: {process.stderr}")
                        continue
                    
                    # 결과 이미지 확인
                    result_images = list(factor_output_dir.glob("*.png"))
                    if not result_images:
                        logger.warning(f"요소 {factor_idx} 적용 결과 이미지를 찾을 수 없습니다")
                        continue
                    
                    logger.info(f"요소 {factor_idx}, 강도 {strength} 적용 완료")
            
            # 적어도 하나의 요소가 성공적으로 적용되었는지 확인
            any_result = False
            for factor_idx in factors_to_test:
                for strength in strengths:
                    factor_output_dir = self.results_dir / f"factor_{factor_idx}_strength_{strength}"
                    result_images = list(factor_output_dir.glob("*.png"))
                    if result_images:
                        any_result = True
                        break
                if any_result:
                    break
            
            if not any_result:
                logger.error("모든 요소 적용이 실패했습니다")
                return False
            
            logger.info("스타일 요소 적용 테스트 완료")
            self.results["factor_application_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"스타일 요소 적용 테스트 중 오류 발생: {e}")
            return False
    
    def test_style_mixing(self):
        """스타일 믹싱 테스트"""
        try:
            logger.info("스타일 믹싱 테스트 중...")
            
            # 두 개의 시드로 이미지 생성
            seed1 = self.seed
            seed2 = self.seed + 1
            
            # 이미지 생성
            seeds_dir = self.images_dir / "seeds"
            os.makedirs(seeds_dir, exist_ok=True)
            
            cmd_generate = [
                "python", str(self.stylegan_dir / "generate.py"),
                f"--network={self.model_path}",
                f"--seeds={seed1},{seed2}",
                f"--outdir={seeds_dir}"
            ]
            
            logger.info("스타일 믹싱용 이미지 생성 중...")
            process = subprocess.run(
                cmd_generate, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"이미지 생성 실패: {process.stderr}")
                return False
            
            # 스타일 믹싱 실행
            
            mixing_dir = self.results_dir / "style_mixing"
            os.makedirs(mixing_dir, exist_ok=True)
            
            cmd_mixing = [
                "python", str(self.stylegan_dir / "style_mixing.py"),
                f"--network={self.model_path}",
                f"--rows={seed1}",
                f"--cols={seed2}",
                "--styles=0-6",
                f"--outdir={mixing_dir}"
            ]
            
            logger.info("스타일 믹싱 실행 중...")
            process = subprocess.run(
                cmd_mixing, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.error(f"스타일 믹싱 실패: {process.stderr}")
                return False
            
            # 결과 이미지 확인
            result_image = mixing_dir / "grid.png"
            if not result_image.exists():
                logger.error("스타일 믹싱 결과 이미지를 찾을 수 없습니다")
                return False
            
            logger.info("스타일 믹싱 테스트 완료")
            self.results["style_mixing_test"] = True
            return True
            
        except Exception as e:
            logger.error(f"스타일 믹싱 테스트 중 오류 발생: {e}")
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        try:
            logger.info("=== StyleGAN2-ADA 통합 테스트 시작 ===")
            start_time = time.time()
            
            # 각 테스트 단계 실행
            env_check = self.check_environment()
            if not env_check:
                logger.error("환경 확인 실패! 테스트를 중단합니다.")
                return False
            
            image_prep = self.prepare_test_image()
            if not image_prep:
                logger.error("테스트 이미지 준비 실패! 테스트를 중단합니다.")
                return False
            
            projection = self.test_projection()
            if not projection:
                logger.warning("투영 테스트 실패! 다음 테스트로 진행합니다.")
            
            factorization = self.test_closed_form_factorization()
            if not factorization:
                logger.warning("스타일 요소 추출 테스트 실패! 다음 테스트로 진행합니다.")
            
            # 투영과 요소 추출이 모두 성공했을 때만 요소 적용 테스트 진행
            if projection and factorization:
                factor_application = self.test_apply_factor()
                if not factor_application:
                    logger.warning("스타일 요소 적용 테스트 실패! 다음 테스트로 진행합니다.")
            else:
                logger.warning("이전 단계 실패로 스타일 요소 적용 테스트를 건너뜁니다.")
            
            # 스타일 믹싱 테스트는 독립적으로 실행 가능
            style_mixing = self.test_style_mixing()
            if not style_mixing:
                logger.warning("스타일 믹싱 테스트 실패!")
            
            # 전체 성공 여부 평가
            success_count = sum([
                self.results["environment_check"],
                self.results["test_image_preparation"],
                self.results["projection_test"],
                self.results["factor_extraction_test"],
                self.results["factor_application_test"],
                self.results["style_mixing_test"]
            ])
            
            total_tests = 6
            success_rate = (success_count / total_tests) * 100
            
            self.results["overall_success"] = success_rate >= 50  # 50% 이상 성공 시 전체 성공으로 간주
            self.results["success_rate"] = f"{success_rate:.1f}%"
            self.results["execution_time"] = f"{(time.time() - start_time):.1f}초"
            
            # 결과 저장
            with open(self.test_dir / "test_results.json", "w") as f:
                json.dump(self.results, f, indent=2)
            
            # 결과 요약 출력
            logger.info("=== 테스트 결과 요약 ===")
            logger.info(f"환경 확인: {'성공' if self.results['environment_check'] else '실패'}")
            logger.info(f"테스트 이미지 준비: {'성공' if self.results['test_image_preparation'] else '실패'}")
            logger.info(f"투영 테스트: {'성공' if self.results['projection_test'] else '실패'}")
            logger.info(f"스타일 요소 추출 테스트: {'성공' if self.results['factor_extraction_test'] else '실패'}")
            logger.info(f"스타일 요소 적용 테스트: {'성공' if self.results['factor_application_test'] else '실패'}")
            logger.info(f"스타일 믹싱 테스트: {'성공' if self.results['style_mixing_test'] else '실패'}")
            logger.info(f"성공률: {self.results['success_rate']}")
            logger.info(f"실행 시간: {self.results['execution_time']}")
            logger.info(f"전체 결과: {'성공' if self.results['overall_success'] else '실패'}")
            logger.info(f"테스트 결과 디렉토리: {self.test_dir}")
            
            return self.results["overall_success"]
            
        except Exception as e:
            logger.error(f"테스트 실행 중 오류 발생: {e}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description='StyleGAN2-ADA 통합 테스트')
    parser.add_argument('--stylegan_dir', type=str, required=True,
                        help='StyleGAN2-ADA-PyTorch 디렉토리 경로')
    parser.add_argument('--model', type=str, required=True,
                        help='StyleGAN2 모델 파일 경로 (.pkl)')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='테스트 결과 저장 디렉토리')
    parser.add_argument('--test_image', type=str, default='',
                        help='테스트에 사용할 이미지 경로 (선택사항)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    return parser.parse_args()

def main():
    args = parse_args()
    tester = StyleGANTester(args)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()