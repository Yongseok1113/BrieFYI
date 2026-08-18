"""기사 본문 청킹, embedding, GLiNER2 4-Layer indexing, retrieval을 제공하는 RAG 패키지.

rag/를 뼈대로 하고, rag_experiment/에서 검증된 reranker.py(cross-encoder 재정렬)와
agent_tool.py(LLM tool-use 검색 래퍼)를 이식했다. rag/, rag_experiment/는 원본 보존을
위해 그대로 두고 손대지 않는다(추후 삭제 예정).

    content.py    기사 URL 본문 수집 + 텍스트 구성 + BGE token 청킹
    embed.py      BGE-M3 dense embedding (Hugging Face Inference API)
    extract.py    GLiNER2 Category/Domain/Entity와 구조화 Event 추출
    taxonomy.py   4-Layer 고정 taxonomy와 추출 설정
    db.py         RAG 전용 SQL 전부
    indexer.py    indexing stage (chunk/embedding/4-Layer 저장)
    retriever.py  vector/text/hybrid 검색
    pipeline.py   수집 -> indexing 오케스트레이션
    cli.py        단일 CLI 진입점
    reranker.py   cross-encoder 재정렬 (rag_experiment/rag/reranker.py 이식)
    agent_tool.py LLM tool-use 검색 래퍼 (rag_experiment/rag/agent_search.py 이식)
    eval.py       recall@k/MRR 평가 (rag_experiment의 eval_*.py 계열 이식/통합)
"""

__version__ = "0.1.0"
