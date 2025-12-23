#!/bin/bash

# 쉘 스크립트 목적: ShapeLLM 기반 대화 CLI 실행 (RAG 통합된 상태)
# 실행 전제: 현재 경로에 cli.py, rag_fetcher.py, rag_output.txt 있어야 함
# 가상환경: shapellm (LLM 실행용), rag_env (rag_fetcher.py subprocess 용)

# ⛳ 가상환경 설정 (여기선 shapellm 환경에서 실행한다고 가정)

# ✅ 실행 로그 출력
echo "\n🚀 ShapeLLM RAG 대화 세션 시작"
echo "-----------------------------------"

# ✅ CLI 실행 (rag_fetcher는 subprocess로 자동 호출됨)
python cli2.py \
  --model-path qizekun/ShapeLLM_13B_general_v1.0 \
  --pts-file assets/SW_Scaffold_8192.npy

# 🎉 종료 메시지
echo "\n🛑 대화 세션 종료됨. 고생했어!"