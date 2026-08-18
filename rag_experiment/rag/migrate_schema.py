"""
RAG 인덱스 구조 개선 - 스키마 마이그레이션

rag_documents 테이블에 category, domain, entities, event
이렇게 4개의 새 컬럼을 추가한다.
검색 조건으로 쓸 4개 컬럼을 새로 추가하는 것.
"""

# psycopg2: 파이썬에서 PostgreSQL(데이터베이스)에 접속하기 위한 라이브러리
import psycopg2


# ==========================================
# 1. DB 접속 정보
# ==========================================
# host: DB가 실행 중인 위치 (localhost = 내 컴퓨터)
# port: docker-compose.yml에서 5433으로 열어놓은 포트
# dbname/user/password: docker-compose.yml에 설정된 값
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "rag_test",
    "user": "rag_user",
    "password": "rag_password",
}


# ==========================================
# 2. 마이그레이션 실행 함수
# ==========================================

def migrate():
    """
    실제로 DB에 접속해서 컬럼을 추가하는 함수.
    """

    # DB_CONFIG에 있는 정보로 실제 데이터베이스에 접속
    conn = psycopg2.connect(**DB_CONFIG)

    # cursor: DB에 SQL 명령을 보내고 결과를 받는 통로
    cur = conn.cursor()

    # try/except/finally 구조:
    # - try 안에서 에러가 나면 except로 가서 롤백(변경 취소)
    # - 성공하든 실패하든 finally는 무조건 실행돼서 연결을 닫음
    try:

        # --------------------------------------
        # ALTER TABLE: 이미 있는 테이블에 컬럼을 추가하는 SQL 명령어
        # ADD COLUMN IF NOT EXISTS: "그 컬럼이 이미 있으면 아무것도 안 하고,
        #                            없으면 새로 추가해라"
        #   → 이 IF NOT EXISTS 덕분에 스크립트를 실수로 두 번 돌려도
        #     에러 없이 안전하다 (이미 있으니 그냥 넘어감)
        #
        # 각 컬럼 타입:
        # - category TEXT   → "경제", "기술" 같은 문자열 하나
        # - domain TEXT     → "AI", "반도체" 같은 문자열 하나
        # - entities TEXT[] → 문자열 여러 개를 담는 배열.
        #                     예: ["삼성전자", "NVIDIA"] 이런 식으로 여러 개 저장 가능
        # - event TEXT      → "투자", "실적" 같은 문자열 하나
        cur.execute(
            """
            ALTER TABLE rag_documents
                ADD COLUMN IF NOT EXISTS category TEXT,
                ADD COLUMN IF NOT EXISTS domain TEXT,
                ADD COLUMN IF NOT EXISTS entities TEXT[],
                ADD COLUMN IF NOT EXISTS event TEXT
            """
        )

        # --------------------------------------
        # 인덱스(INDEX) 생성
        #
        # 인덱스: 책의 "찾아보기" 페이지 같은 것
        # 인덱스를 만들어두면 바로 찾아갈 수 있어서 훨씬 빠르다.
        #
        # category, domain, event는 "값이 같은지"만 비교하는
        # 일반적인 컬럼이라서 기본 인덱스(B-tree)로 충분
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_category "
            "ON rag_documents (category)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_domain "
            "ON rag_documents (domain)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_event "
            "ON rag_documents (event)"
        )

        # entities는 배열(TEXT[])이라서 일반 인덱스로는 검색이 안 되고
        # GIN이라는 특수한 인덱스 방식을 써야
        # "이 배열 안에 'NVIDIA'가 들어있는 행 찾아줘" 같은 검색이 빨라진다.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_entities "
            "ON rag_documents USING GIN (entities)"
        )

        # --------------------------------------
        # commit(): 위에서 실행한 변경사항들을 DB에 실제로 저장(확정)한다.
        # commit()을 안 하면 변경사항이 임시 상태로만 남고 실제 반영이 안 됨.
        conn.commit()

        print("마이그레이션 완료: category, domain, entities, event 컬럼 추가됨")

    except Exception:

        # 중간에 에러가 나면 지금까지 한 변경을 전부 취소(rollback)한다.
        # DB가 애매하게 반쯤 바뀐 상태로 남는 걸 방지.
        conn.rollback()

        # 에러를 다시 던져서, 이 스크립트를 실행한 사람이
        # 콘솔에서 무슨 에러인지 볼 수 있게 한다.
        raise

    finally:

        # 성공하든 실패하든 마지막에 DB 연결을 정리(닫기)한다.
        # 연결을 안 닫으면 나중에 "연결이 너무 많다" 같은 문제가 생길 수 있음.
        cur.close()
        conn.close()


# ==========================================
# 3. 이 파일을 직접 실행했을 때만 동작
# ==========================================
# 다른 파일에서 이 파일을 import만 할 때는 migrate()가 자동 실행되지 않고,
# "python -m rag.migrate_schema" 처럼 이 파일을 직접 실행할 때만 동작한다.
if __name__ == "__main__":
    migrate()