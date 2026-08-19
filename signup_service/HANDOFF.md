구독 신청 페이지 — 진행 상황 (demo 브랜치)
지금 상태
web/brief-signup.html: 완성된 신청 화면 (경제/기술/금융/산업/스포츠/정치 카테고리)
signup_service/: 신청 데이터를 저장하고 이메일을 보내는 백엔드 (FastAPI)
db/schema.sql: subscribers 테이블 추가됨 (맨 아래 참고)

로컬(개발자 컴퓨터)에서는 완전히 동작 확인됨:

실제 팀 Postgres에 구독자 저장 ✅
신청 시 환영 메일 발송 ✅ (Resend API)
구독 취소(DELETE) ✅


지금은 signup_service가 개발자 로컬 컴퓨터에서만 실행 중이라, 다른 사람이 사이트에 들어가서 신청하면 "서버에 연결할 수 없어요" 에러가 뜬다. 실제 서비스로 쓰려면 아래 작업이 남아있다.

필요한 것
signup_service를 실제로 계속 켜져 있는 서버(Render, Railway 등)에 배포
그 서버가 접속할 수 있는 클라우드 DB 필요 (지금 팀 Postgres는 로컬 Docker 안에서만 도는 상태라 외부에서 접속 불가)
web/brief-signup.html의 API_BASE 상수를, 배포된 서버 주소로 변경
js
   const API_BASE = 'http://127.0.0.1:8000';  // -> https://briefyi.netlify.app/ 실제 배포 주소로 교체
Netlify를 GitHub demo 브랜치에 연결 (Publish directory: web) — 현재 담당자 GitHub 계정 권한 문제로 보류 중, 저장소 소유자 승인 필요
참고 — 팀에 이미 있는 .github/workflows/daily-digest.yml은 재사용 불가


로컬에서 테스트하려면
powershell
pip install -r requirements.txt -r signup_service/requirements.txt
docker compose up -d db
uvicorn signup_service.app:app --reload --port 8000

.env에 필요한 값: DATABASE_URL(또는 POSTGRES_*), RESEND_API_KEY, EMAIL_FROM, EMAIL_TO