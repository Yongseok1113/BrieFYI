"""기사 본문 청킹, embedding, GLiNER2 4-Layer indexing, retrieval을 제공하는 RAG 패키지.

구성은 develop 브랜치의 `data_pipeline`을 따른다 — SQL은 `db.py` 한 곳, 실행 진입점은
`cli.py` 한 곳에 모으고 나머지는 기능별 모듈로 나눈다. 다만 data_pipeline과 달리 rag는
별도 Docker 이미지가 아니라 메인 앱과 같은 이미지·같은 DB를 쓰므로, DB 연결
(`db.db.get_conn`)과 뉴스 수집 도구(`tools.news_fetch`)는 레포 루트 것을 그대로 재사용하고
RAG 고유 로직(본문 추출, 청킹, GLiNER2 추출, 검색)만 이 패키지 안에 둔다.

    content.py    기사 URL 본문 수집 + 텍스트 구성 + BGE token 청킹
    embed.py      BGE-M3 dense embedding (Hugging Face Inference API)
    extract.py    GLiNER2 Category/Domain/Entity와 구조화 Event 추출
    taxonomy.py   4-Layer 고정 taxonomy와 추출 설정
    db.py         RAG 전용 SQL 전부
    indexer.py    indexing stage (chunk/embedding/4-Layer 저장)
    retriever.py  vector/text/hybrid 검색
    pipeline.py   수집 -> indexing 오케스트레이션
    cli.py        단일 CLI 진입점
"""

__version__ = "0.1.0"
