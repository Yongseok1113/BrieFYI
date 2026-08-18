import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5433,
    dbname="rag_test",
    user="rag_user",
    password="rag_password",
)

cur = conn.cursor()

cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'rag_documents'
    ORDER BY ordinal_position
""")

print("=== 컬럼 구조 ===")
for row in cur.fetchall():
    print(row)

cur.execute("SELECT COUNT(*) FROM rag_documents")
print("\n전체 문서:", cur.fetchone()[0])

cur.execute("""
    SELECT COUNT(*)
    FROM rag_documents
    WHERE embedding IS NOT NULL
""")
print("벡터 존재 문서:", cur.fetchone()[0])

cur.close()
conn.close()