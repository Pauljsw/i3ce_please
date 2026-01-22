# ShapeLLM Scaffold Missing Detection - Complete Workflow Guide

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Phase 1] Data Generation (합성 데이터)                                     │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 2] Stage 1 Training (Feature Alignment)                            │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 3] Stage 2 Training (Instruction Tuning)                           │
│      │                                                                      │
│      ▼                                                                      │
│  [Phase 4] Inference (Generate Predictions)                                │
│      │                                                                      │
│      ├───────────────────────────────────┐                                 │
│      ▼                                   ▼                                 │
│  [Phase 5] Synthetic Evaluation     [Phase 6] Real Data Validation         │
│      │                                   │                                 │
│      └───────────────────────────────────┘                                 │
│                      │                                                      │
│                      ▼                                                      │
│  [Phase 7] Visualization (Paper Figures)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Structure Overview

### 데이터 파일 구조

```
평가용 데이터 3종:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 파일                    │ 주요 필드                      │ 용도            │
├─────────────────────────────────────────────────────────────────────────────┤
│ test_questions.jsonl    │ question_id, text (질문)       │ 모델 입력       │
│ test_gt.jsonl           │ question_id, text (정답),      │ 평가 GT         │
│                         │ label, task_type, bboxes       │                 │
│ predictions.jsonl       │ question_id, text (모델출력)   │ 평가 대상       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### task_type 종류 및 의미

| task_type | 질문 예시 | Yes 의미 | No 의미 |
|-----------|----------|----------|---------|
| `missing_detection_summary` | "Are there any missing components?" | 결함 있음 | 결함 없음 |
| `missing_detection_floor` | "Are there missing components on floor 2?" | 해당 층 결함 | 해당 층 정상 |
| `missing_detection_specific` | "Is there a platform at floor 2, bay 1?" | **존재함** | **누락됨** |

> **주의**: `missing_detection_specific`은 Yes/No 의미가 반대입니다!

---

## Phase 1: Data Generation

### When to Run
- **First time setup**
- When changing scaffold configurations
- When increasing dataset size

### Command
```bash
cd /home/user/i3ce_shape

# Quick test (1000 scenes)
python -m tools.scaffold_data_pipeline.main \
    --num-scenes 1000 \
    --output-dir ./playground/data/shapellm/scaffold_v2

# Full dataset (3000+ scenes) with ablation data
python -m tools.scaffold_data_pipeline.main \
    --num-scenes 3000 \
    --output-dir ./playground/data/shapellm/scaffold_v2 \
    --with-ablation
```

### Output Files
```
playground/data/shapellm/scaffold_v2/
├── pcs/                           # Point cloud files (.npy)
│   ├── scaffold_00000.npy
│   └── ...
├── stage1_caption_train.json      # Stage 1 학습용 (캡션)
├── stage1_caption_val.json        # Stage 1 검증용
├── stage2_sft_train.json          # Stage 2 학습용 (QA)
├── stage2_sft_val.json            # Stage 2 검증용
├── test_questions.jsonl           # 테스트 질문
├── test_gt.jsonl                  # 테스트 정답
├── ablation_noise/                # Ablation: 랜덤 노이즈
├── ablation_shuffled/             # Ablation: 점 순서 셔플
└── ablation_cross/                # Ablation: 다른 비계 매칭
```

### Verification
```bash
# Check data counts
wc -l playground/data/shapellm/scaffold_v2/*.json
head -c 500 playground/data/shapellm/scaffold_v2/stage1_caption_train.json
```

---

## Phase 2: Stage 1 Training (Feature Alignment)

### Purpose
- mm_projector 학습 (점군 특징 → LLM 임베딩 정렬)
- LLM backbone은 frozen

### When to Run
- After data generation
- When retraining from scratch

### Command
```bash
# Terminal 1: Training
bash scripts/pretrain_scaffold_stage1.sh

# Terminal 2: Tensorboard (optional, 실시간 모니터링)
tensorboard --logdir ./logs --port 6006
# Browser: http://localhost:6006
```

### Duration
- 5 epochs, ~2100 samples
- Estimated: 30-60 minutes on 4x RTX 3090

### Output Files
```
checkpoints/scaffold-stage1-feature-alignment/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── mm_projector.bin               # ⭐ Stage 2에서 사용
├── trainer_state.json             # 학습 로그
└── config.json

logs/scaffold-stage1-feature-alignment/
└── events.out.tfevents.*          # Tensorboard 로그
```

### Verification
```bash
# Check checkpoint exists
ls -la checkpoints/scaffold-stage1-feature-alignment/

# Check training loss decreased
cat checkpoints/scaffold-stage1-feature-alignment/trainer_state.json | python -c "
import json, sys
data = json.load(sys.stdin)
logs = [l for l in data['log_history'] if 'loss' in l]
print(f'Initial loss: {logs[0][\"loss\"]:.4f}')
print(f'Final loss: {logs[-1][\"loss\"]:.4f}')
"
```

---

## Phase 3: Stage 2 Training (Instruction Tuning)

### Purpose
- LoRA로 LLM fine-tuning
- 부재 탐지 태스크 학습

### When to Run
- After Stage 1 completion
- Requires mm_projector.bin from Stage 1

### Command
```bash
# Terminal 1: Training
bash scripts/finetune_scaffold_stage2.sh

# Terminal 2: Tensorboard
tensorboard --logdir ./logs --port 6006
```

### Duration
- 3 epochs, ~10000 samples
- Estimated: 2-4 hours on 4x RTX 3090

### Output Files
```
checkpoints/scaffold-stage2-instruction-tuning/
├── checkpoint-100/
├── checkpoint-200/
├── ...
├── adapter_model.bin              # LoRA weights
├── adapter_config.json
├── trainer_state.json
└── config.json

logs/scaffold-stage2-instruction-tuning/
└── events.out.tfevents.*
```

---

## Phase 4: Inference (Generate Predictions)

### Purpose
- 테스트 데이터에 대한 모델 예측 생성

### When to Run
- After Stage 2 completion
- Before evaluation

### Command
```bash
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/pcs \
    --answers-file ./outputs/test_answers.jsonl \
    --temperature 0.2
```

### For Ablation Studies
```bash
# Noise ablation
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/ablation_noise/questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/ablation_noise/pcs \
    --answers-file ./outputs/ablation_noise_answers.jsonl

# Repeat for other ablations...
```

### Output
```
outputs/
├── test_answers.jsonl             # 메인 예측
├── ablation_noise_answers.jsonl   # Ablation 예측들
├── ablation_shuffled_answers.jsonl
└── ablation_cross_answers.jsonl
```

---

## Phase 5: Synthetic Data Evaluation

### Purpose
- 합성 테스트 데이터에 대한 종합 평가
- Data Leakage 검증 (Ablation)
- 논문용 메트릭 생성

### 5A. Rule-based Evaluation (기본)

```bash
# Full evaluation with ablation verification
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --ablation-dir ./playground/data/shapellm/scaffold_v2 \
    --output-dir ./evaluation_results

# Quick evaluation (no ablation)
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --quick \
    --output-dir ./evaluation_results
```

### 5B. GPT-based Evaluation (선택, 권장)

GPT-4/3.5를 judge로 사용한 의미적 평가

```bash
# Detailed evaluation (5 categories)
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --model gpt-4

# Simple scoring (faster, cheaper)
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --model gpt-3.5-turbo \
    --simple

# Test with limited samples first
python tools/gpt_evaluate_scaffold.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --questions ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --output-dir ./evaluation_results \
    --openai-key YOUR_OPENAI_API_KEY \
    --max-samples 50
```

### Output
```
evaluation_results/
├── evaluation_report.json         # Rule-based 결과 (JSON)
├── results_table.tex              # LaTeX 테이블 (논문용)
├── gpt_eval_results.jsonl         # GPT 개별 샘플 점수
└── gpt_eval_summary.json          # GPT 카테고리별 평균
```

### Metrics Explained

| Category | Metric | Description |
|----------|--------|-------------|
| **Binary** | Accuracy | Yes/No 정답률 |
| | Precision | TP / (TP + FP) |
| | Recall | TP / (TP + FN) |
| | F1 | 2 × P × R / (P + R) |
| **Component** | Type Accuracy | 부재 종류별 정확도 |
| **Count** | Exact Match | 개수 정확히 맞춤 |
| | ±1 Tolerance | 1개 오차 허용 |
| **BBox** | IoU@0.5 | 50% 이상 겹침 비율 |
| | IoU@0.25 | 25% 이상 겹침 비율 |
| | Detection Rate | GT 대비 탐지율 |

### GPT Evaluation Categories

| Category | Description | Scoring |
|----------|-------------|---------|
| Binary Detection | Yes/No 정답 여부 | 100=정답, 0=오답 (중간값 없음) |
| Component Type | 부재 종류 식별 | 100=완벽, 50=부분, 0=오류 |
| Count Accuracy | 개수 정확도 | 100=정확, 70=±1, 40=±2 |
| Location Description | 위치 설명 정확도 | 100=정확, 50=부분 |
| Overall Quality | 전체 품질 | 종합 평가 |

### Cost Estimation (GPT)
- GPT-3.5-turbo: ~$0.001-0.002 per sample
- GPT-4: ~$0.03-0.05 per sample
- 1000 samples with GPT-4 ≈ $30-50

---

## Phase 6: Real Data Validation (Sim2Real)

### Purpose
- 합성 데이터로 학습한 모델이 **실제 점군**에서도 작동하는지 검증
- 학술적 타당성의 핵심 (Sim2Real generalization)

### 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  수작업 (CloudCompare, 1-2시간)                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. 실제 비계 점군 파일 선택 (1-2개)                                          │
│  2. CloudCompare에서 각 부재의 bbox 좌표 기록                                 │
│  3. CSV 파일로 저장                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  자동화 (Python 스크립트)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. prepare_real_data.py 실행                                               │
│  2. 점군 정규화 + 부재 제거 → 결함 점군 생성                                  │
│  3. Ground Truth 자동 생성                                                   │
│  4. 모델 Inference → 평가                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6A. 수작업: 부재 좌표 CSV 작성

**CloudCompare에서 부재 bbox 좌표 얻는 방법:**

1. 점군 파일 열기
2. Segment Tool (가위 아이콘)로 부재 하나 선택
3. Properties → Bounding Box 확인:
   - Min corner: X, Y, Z
   - Max corner: X, Y, Z
4. 값을 CSV에 기록

**CSV 파일 형식 (`scaffold_A_components.csv`):**

```csv
component_id,type,x_min,x_max,y_min,y_max,z_min,z_max
c1,vertical,-5.23,-4.18,0.12,0.98,0.00,24.56
c2,vertical,4.12,5.07,0.15,1.01,0.00,24.56
c3,horizontal,-5.23,5.07,0.50,0.65,12.00,12.50
c4,platform,-5.20,5.04,0.12,8.95,11.80,12.20
```

**컬럼 설명:**

| 컬럼 | 출처 | 설명 |
|------|------|------|
| `component_id` | 자유 작명 | 부재 구분용 (c1, v1, part_a 등) |
| `type` | 부재 종류 | `vertical`, `horizontal`, `platform` 중 택1 |
| `x_min`, `x_max` | CloudCompare | Bounding Box X 범위 |
| `y_min`, `y_max` | CloudCompare | Bounding Box Y 범위 |
| `z_min`, `z_max` | CloudCompare | Bounding Box Z 범위 |

> **참고**: floor/bay 정보는 필요 없습니다. bbox만으로 평가 가능합니다.

### 6B. 자동화: 결함 점군 및 라벨 생성

```bash
# 필요 패키지 (다른 컴퓨터에서 실행 시)
pip install numpy open3d   # .ply 파일용
# 또는
pip install numpy laspy    # .las 파일용

# 실행
python tools/prepare_real_data.py \
    --input-pointcloud ./scaffold_A.ply \
    --components-csv ./scaffold_A_components.csv \
    --output-dir ./real_validation_data \
    --scaffold-id scaffold_A

# 복합 결함 포함 (2개 동시 제거)
python tools/prepare_real_data.py \
    --input-pointcloud ./scaffold_A.ply \
    --components-csv ./scaffold_A_components.csv \
    --output-dir ./real_validation_data \
    --scaffold-id scaffold_A \
    --max-combinations 2
```

### 출력 파일

```
real_validation_data/
├── pointclouds/                       # 정규화된 점군들
│   ├── scaffold_A_complete.npy        # 완전체 (baseline)
│   ├── scaffold_A_c1_removed.npy      # c1 제거됨
│   ├── scaffold_A_c2_removed.npy      # c2 제거됨
│   └── ...
├── real_test_questions.jsonl          # 평가용 질문
├── real_test_gt.jsonl                 # Ground Truth
└── generation_log.json                # 생성 로그
```

### 6C. 모델 Inference 및 평가

```bash
# 1. pointclouds/ 폴더를 모델 서버로 복사

# 2. Inference
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./real_validation_data/real_test_questions.jsonl \
    --point-folder ./real_validation_data/pointclouds \
    --answers-file ./outputs/real_test_answers.jsonl \
    --temperature 0.2

# 3. 평가
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/real_test_answers.jsonl \
    --ground-truth ./real_validation_data/real_test_gt.jsonl \
    --quick \
    --output-dir ./evaluation_results/real_data
```

### 실제 데이터 평가 지표

| 지표 | 중요도 | 설명 |
|------|--------|------|
| Binary Accuracy | ⭐⭐⭐ | 결함 유무 판단 정확도 |
| BBox IoU | ⭐⭐⭐ | 위치 파악 정확도 (Data Leakage 방지 핵심) |
| Count Accuracy | ⭐⭐ | 결함 개수 정확도 |
| Component Type | ⭐⭐ | 부재 유형 식별 |

> **핵심**: Binary Accuracy가 높아도 BBox IoU가 낮으면 Data Leakage 가능성!

---

## Phase 7: Visualization (Paper Figures)

### Purpose
- 논문용 그래프 생성
- Loss curves, comparison charts

### When to Run
- After training completion
- After evaluation

### Command
```bash
# Generate all figures
python tools/paper_visualization.py \
    --all \
    --output-dir ./paper_figures

# Specific plots
python tools/paper_visualization.py --loss-curve
python tools/paper_visualization.py --combined
```

### Output
```
paper_figures/
├── training_loss.png              # Loss 곡선
├── training_loss.pdf              # PDF 버전
├── training_loss.svg              # SVG 버전
├── combined_training.png          # Stage 1+2 비교
├── combined_training.pdf
└── training_summary.json          # 수치 요약
```

---

## Quick Reference: Complete Pipeline Commands

```bash
# ==================== FULL PIPELINE ====================

# 1. Data Generation (합성)
python -m tools.scaffold_data_pipeline.main --num-scenes 3000 --with-ablation

# 2. Stage 1 Training
bash scripts/pretrain_scaffold_stage1.sh

# 3. Stage 2 Training
bash scripts/finetune_scaffold_stage2.sh

# 4. Inference (합성 데이터)
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./playground/data/shapellm/scaffold_v2/test_questions.jsonl \
    --point-folder ./playground/data/shapellm/scaffold_v2/pcs \
    --answers-file ./outputs/test_answers.jsonl

# 5. Evaluation (합성 데이터)
python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/test_answers.jsonl \
    --ground-truth ./playground/data/shapellm/scaffold_v2/test_gt.jsonl \
    --output-dir ./evaluation_results

# 6. Real Data Validation (실제 데이터)
python tools/prepare_real_data.py \
    --input-pointcloud ./real_scaffold.ply \
    --components-csv ./real_components.csv \
    --output-dir ./real_validation_data \
    --scaffold-id real_scaffold_A

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/scaffold-stage2-instruction-tuning \
    --model-base qizekun/ShapeLLM_7B_gapartnet_v1.0 \
    --question-file ./real_validation_data/real_test_questions.jsonl \
    --point-folder ./real_validation_data/pointclouds \
    --answers-file ./outputs/real_test_answers.jsonl

python tools/evaluate_scaffold_rigorous.py \
    --predictions ./outputs/real_test_answers.jsonl \
    --ground-truth ./real_validation_data/real_test_gt.jsonl \
    --quick \
    --output-dir ./evaluation_results/real_data

# 7. Visualization
python tools/paper_visualization.py --all --output-dir ./paper_figures
```

---

## Troubleshooting

### Stage 1 checkpoint not found
```bash
# Check if mm_projector.bin exists
ls checkpoints/scaffold-stage1-feature-alignment/mm_projector.bin

# If not, check for checkpoint directories
ls checkpoints/scaffold-stage1-feature-alignment/checkpoint-*/
```

### CUDA Out of Memory
```bash
# Reduce batch size in training scripts
# Stage 1: per_device_train_batch_size: 2 → 1
# Stage 2: per_device_train_batch_size: 4 → 2
```

### Low BBox IoU
- Check bbox format in model outputs
- Increase training epochs
- Review if questions have proper bbox context

### High Binary Accuracy but Low IoU (Data Leakage)
- Run ablation studies to verify
- Check question templates are uniform
- Ensure model isn't using text shortcuts

### Real Data: No points removed
- Check bbox coordinates match the point cloud coordinate system
- Verify CloudCompare export settings
- Try slightly larger bbox margins

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Data Generation | 5-30 min | None |
| Stage 1 Training | 30-60 min | Data |
| Stage 2 Training | 2-4 hours | Stage 1 |
| Inference (Synthetic) | 30-60 min | Stage 2 |
| Evaluation (Synthetic) | 5-10 min | Inference |
| Real Data Prep | 1-2 hours | CloudCompare (수작업) |
| Inference (Real) | 10-20 min | Real data prep |
| Evaluation (Real) | 5 min | Real inference |
| Visualization | 1-2 min | Training logs |

**Total: ~5-7 hours** (including real data validation)

---

## File Reference

### 주요 스크립트

| 스크립트 | 용도 |
|----------|------|
| `tools/scaffold_data_pipeline/main.py` | 합성 데이터 생성 |
| `scripts/pretrain_scaffold_stage1.sh` | Stage 1 훈련 |
| `scripts/finetune_scaffold_stage2.sh` | Stage 2 훈련 |
| `tools/evaluate_scaffold_rigorous.py` | Rule-based 평가 |
| `tools/gpt_evaluate_scaffold.py` | GPT-based 평가 |
| `tools/prepare_real_data.py` | 실제 데이터 전처리 |
| `tools/paper_visualization.py` | 논문 시각화 |

### 데이터 파일 포맷

**test_questions.jsonl:**
```json
{"question_id": "scaffold_00001_q1", "point": "scaffold_00001.npy", "text": "Are there any missing components?", "category": "scaffold"}
```

**test_gt.jsonl:**
```json
{"question_id": "scaffold_00001_q1", "point": "scaffold_00001.npy", "text": "Yes, missing components detected (2 total):...", "label": "Yes", "bboxes": [...], "task_type": "missing_detection_summary"}
```

**predictions.jsonl:**
```json
{"question_id": "scaffold_00001_q1", "text": "Yes, there are 2 missing components...", "model_id": "shapellm-scaffold"}
```
