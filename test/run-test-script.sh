#!/bin/bash
# StyleGAN2-ADA 통합 테스트 실행 스크립트

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로고 출력
echo -e "${BLUE}"
echo "========================================================"
echo "      StyleGAN2-ADA 통합 테스트 실행 스크립트           "
echo "========================================================"
echo -e "${NC}"

# 필요한 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(/home/elicer/stylegan2-ada-pytorch "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test/integrated_test.py"
OUTPUT_DIR="$SCRIPT_DIR/test/test_results"

# StyleGAN2-ADA-PyTorch 디렉토리 경로 확인
if [ -z "$STYLEGAN_DIR" ]; then
    echo -e "${YELLOW}StyleGAN2-ADA-PyTorch 디렉토리 경로를 입력하세요:${NC}"
    read STYLEGAN_DIR
    
    if [ ! -d "$STYLEGAN_DIR" ]; then
        echo -e "${RED}오류: 지정한 경로가 존재하지 않습니다: $STYLEGAN_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$STYLEGAN_DIR/generate.py" ] || [ ! -f "$STYLEGAN_DIR/projector.py" ]; then
        echo -e "${RED}오류: 지정한 경로가 StyleGAN2-ADA-PyTorch 디렉토리가 아닌 것 같습니다.${NC}"
        exit 1
    fi
fi

# 모델 파일 경로 확인
if [ -z "$MODEL_PATH" ]; then
    echo -e "${YELLOW}StyleGAN2 모델 파일(.pkl) 경로를 입력하세요:${NC}"
    read MODEL_PATH
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}오류: 지정한 모델 파일이 존재하지 않습니다: $MODEL_PATH${NC}"
        exit 1
    fi
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 테스트 이미지 경로 (선택 사항)
echo -e "${YELLOW}테스트 이미지 경로를 입력하세요 (건너뛰려면 Enter):${NC}"
read TEST_IMAGE

# 시스템 정보 출력
echo -e "${BLUE}시스템 정보:${NC}"
echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -n 1 | cut -d':' -f2 | sed 's/^ //')"
echo "메모리: $(free -h | grep Mem | awk '{print $2}')"

# GPU 정보 출력
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "CUDA 버전: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
else
    echo -e "${YELLOW}경고: nvidia-smi 명령을 찾을 수 없습니다. GPU 정보를 확인할 수 없습니다.${NC}"
fi

# 파이썬 및 필요 라이브러리 확인
echo -e "${BLUE}환경 확인:${NC}"
echo "Python 버전: $(python --version 2>&1)"
echo "PyTorch 버전: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA 가용성: $(python -c 'import torch; print(f"사용 가능: {torch.cuda.is_available()}, 장치 수: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")' 2>&1)"

# 테스트 실행
echo -e "\n${GREEN}테스트 실행 중...${NC}"
echo "명령: python $TEST_SCRIPT --stylegan_dir=$STYLEGAN_DIR --model=$MODEL_PATH --output_dir=$OUTPUT_DIR --test_image=$TEST_IMAGE"

python "$TEST_SCRIPT" --stylegan_dir="$STYLEGAN_DIR" --model="$MODEL_PATH" --output_dir="$OUTPUT_DIR" ${TEST_IMAGE:+--test_image="$TEST_IMAGE"}

# 실행 결과 확인
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}테스트가 성공적으로 완료되었습니다!${NC}"
else
    echo -e "\n${RED}테스트 중 오류가 발생했습니다. 로그 파일을 확인하세요.${NC}"
fi

# 테스트 결과 디렉토리 찾기
LATEST_TEST_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "test_*" | sort -r | head -n 1)

if [ -n "$LATEST_TEST_DIR" ]; then
    echo -e "${BLUE}테스트 결과 디렉토리: $LATEST_TEST_DIR${NC}"
    
    # 결과 파일 확인
    RESULTS_FILE="$LATEST_TEST_DIR/test_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo -e "${GREEN}테스트 결과 요약:${NC}"
        python -c "import json; f=open('$RESULTS_FILE'); d=json.load(f); print(f'성공률: {d.get(\"success_rate\", \"N/A\")}'); print(f'실행 시간: {d.get(\"execution_time\", \"N/A\")}'); print(f'전체 결과: {\"성공\" if d.get(\"overall_success\", False) else \"실패\"}'); f.close()"
    fi
fi

echo -e "${BLUE}자세한 내용은 로그 파일 stylegan_test.log를 확인하세요.${NC}"
echo -e "${GREEN}테스트 완료!${NC}"

exit $EXIT_CODE