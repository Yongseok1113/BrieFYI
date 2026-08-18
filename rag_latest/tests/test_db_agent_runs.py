import unittest
import uuid

from db.db import get_conn, init_db


class SummarizeAgentRunsSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            init_db()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"DB 연결 불가: {exc}")

    def test_테이블에_INSERT하고_조회할_수_있다(self):
        run_id = str(uuid.uuid4())
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO summarize_agent_runs
                   (run_id, query, attempt_number, summary, citations, grounding_passed,
                    grounding_issues, judge_score, judge_reasoning, provider, passed_threshold)
                   VALUES (%s, 'test query', 1, 'test summary', '[]', true, '[]', 85.0,
                           'ok', 'groq', true)""",
                (run_id,),
            )
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM summarize_agent_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        self.assertEqual(row["query"], "test query")
        self.assertTrue(row["passed_threshold"])
        with get_conn() as conn:
            conn.execute("DELETE FROM summarize_agent_runs WHERE run_id = %s", (run_id,))

    def test_judge_score_범위_제약이_동작한다(self):
        import psycopg

        run_id = str(uuid.uuid4())
        with self.assertRaises(psycopg.errors.CheckViolation):
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO summarize_agent_runs
                       (run_id, query, attempt_number, summary, grounding_passed,
                        judge_score, provider, passed_threshold)
                       VALUES (%s, 'q', 1, 's', true, 150.0, 'groq', true)""",
                    (run_id,),
                )


if __name__ == "__main__":
    unittest.main()
