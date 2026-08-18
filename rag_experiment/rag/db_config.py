"""
RAG 공용 DB 접속 정보

여러 파일(indexer.py, retriever.py, migrate_schema.py 등)에
똑같은 DB_CONFIG가 복사돼있던 걸 여기 한 곳으로 모았다.

앞으로 새로 만드는 파일(에이전트 툴, 평가 스크립트)은
이 파일에서 DB_CONFIG를 가져다 쓴다.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_test",
    "user": "rag_user",
    "password": "rag_password",
}