# news-insight-agent (MVP 고정 파이프라인)

GNews 수집 → SQLite 저장 → 요약 → 인사이트/비즈니스 시사점 → 이메일 발송까지, 조건 분기 없이 고정된 순서로 실행되는 LangGraph 파이프라인이다. `mvp-implementation-breakdown.md`의 1단계(#1~#8) 구현체다.

## 구조

```
news-insight-agent/
  config.py              # .env 로드
  main.py                 # CLI 진입점 (#7 실행 트리거, #8 스케줄 대상)
  db/
    schema.sql            # 테이블 정의
    db.py                  # SQLite 헬퍼 (#2)
  tools/
    news_fetch.py          # GNews 수집 (#1)
    llm_client.py           # Anthropic 공용 호출
    summarize.py            # 요약 (#3)
    insight.py               # 인사이트+시사점 (#4)
    email_format.py          # Jinja2 렌더링 (#5)
    email_send.py            # Resend 발송 (#6)
  templates/
    digest_email.html.j2     # 이메일 템플릿
  graph/
    pipeline.py               # LangGraph StateGraph (#7)
  .github/workflows/
    daily-digest.yml          # 스케줄 실행 예시 (#8)
```

## 설정

1. `.env.example`을 `.env`로 복사하고 `GNEWS_API_KEY`, `ANTHROPIC_API_KEY`, `RESEND_API_KEY`, `EMAIL_FROM`, `EMAIL_TO`를 채운다.
2. `pip install -r requirements.txt`

## 실행

```bash
python main.py --keyword "AI" --days 1 --max-results 10
```

첫 실행 시 `DB_PATH` 위치에 SQLite 파일과 테이블이 자동 생성된다.

## 파이프라인 흐름 (고정 순서, 분기 없음)

`fetch_news` → `store_raw` → `summarize` → `extract_insight` → `format_email` → `send_email`

각 노드는 `graph/pipeline.py`의 `PipelineState`(TypedDict)를 입출력으로 공유한다. 노드 하나가 실패하면 `error` 필드에 기록되고 `main.py`가 비정상 종료 코드를 반환한다.

## 스케줄링

`.github/workflows/daily-digest.yml`은 매일 08:00 KST에 `main.py`를 실행하는 GitHub Actions 예시다. 리포지토리 Secrets에 API 키를 등록하면 그대로 동작한다. 서버에서 직접 돌린다면 동일한 명령을 cron에 등록하면 된다.

```
0 8 * * * cd /path/to/news-insight-agent && /usr/bin/python3 main.py >> logs/run.log 2>&1
```

## 다음 확장 (백로그, `mvp-implementation-breakdown.md` 2단계 참고)

Discord 발송(#9), 검증/품질게이트(#10), 기술문서 수집(#11), 중복제거/클러스터링(#12), 실시간 트리거(#13)는 아직 구현하지 않았다. `tools/`에 새 도구를 추가하고 `graph/pipeline.py`에 노드/엣지를 붙이는 방식으로 확장하면 된다. 예를 들어 Discord는 `tools/discord_send.py`를 만들고 `format_email` 다음에 `send_discord` 노드를 병렬로 붙이면 된다. 검증 게이트를 추가할 때 비로소 `add_conditional_edges`로 "재작업 여부"를 LLM이 판단하게 해, 고정 파이프라인에서 에이전틱 파이프라인으로 전환할 수 있다.

## 참고 문서

- GNews Search Endpoint: https://docs.gnews.io/endpoints/search-endpoint
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- Resend API: https://resend.com/docs/api-reference/emails/send-email
