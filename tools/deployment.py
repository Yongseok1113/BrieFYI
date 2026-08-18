import json
from pathlib import Path
from typing import Callable

from db.db import log_send
from tools.email_format import format_email
from tools.email_send import send_email


# 배포 대상 이메일 주소가 들어있는 파일
TARGET_ADDRESS_PATH = Path("target_adress.json")


def load_recipients(
    path: Path = TARGET_ADDRESS_PATH,
) -> list[str]:
    """
    target_adress.json에서 배포 대상 이메일 주소를 읽는다.

    반환값:
        ["example1@gmail.com", "example2@gmail.com"]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"배포 대상 파일을 찾을 수 없습니다: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        recipients = json.load(file)

    # 이메일 목록이 리스트인지 확인
    if not isinstance(recipients, list) or not recipients:
        raise ValueError(
            "배포 대상 이메일 주소가 없습니다."
        )

    # 각각의 이메일 주소 형식 간단 검증
    for recipient in recipients:
        if not isinstance(recipient, str) or "@" not in recipient:
            raise ValueError(
                f"잘못된 이메일 주소입니다: {recipient}"
            )

    return recipients


def validate_deployment_data(
    summaries: list[dict],
    insight: dict,
) -> None:
    """
    이메일로 배포할 데이터가 정상적으로 존재하는지 검증한다.
    """

    if not summaries:
        raise ValueError(
            "배포할 요약 데이터가 없습니다."
        )

    if not insight:
        raise ValueError(
            "배포할 인사이트 데이터가 없습니다."
        )


def prepare_deployment(
    digest_date: str,
    keyword: str,
    summaries: list[dict],
    insight: dict,
) -> dict:
    """
    이메일 배포에 필요한 데이터를 준비한다.

    수행 작업:
    1. 요약/인사이트 데이터 검증
    2. 수신자 목록 조회
    3. JSON 형태의 데이터를 HTML 이메일로 변환
    4. 이메일 제목 생성
    """

    validate_deployment_data(
        summaries,
        insight,
    )

    recipients = load_recipients()

    # 기존 email_format.py를 이용해서
    # 요약 + 인사이트 데이터를 HTML로 변환
    html = format_email(
        digest_date=digest_date,
        keyword=keyword,
        summaries=summaries,
        insight=insight,
    )

    if not html.strip():
        raise ValueError(
            "생성된 이메일 HTML이 비어 있습니다."
        )

    subject = f"[{digest_date}] {keyword} 뉴스&인사이트"

    return {
        "subject": subject,
        "html": html,
        "recipients": recipients,
    }


def deploy_digest(
    digest_id: int,
    digest_date: str,
    keyword: str,
    summaries: list[dict],
    insight: dict,
    sender: Callable = send_email,
) -> dict:
    """
    완성된 뉴스 다이제스트를 이메일로 배포한다.

    수행 작업:
    1. 배포 데이터 준비
    2. 수신자 목록 조회
    3. 수신자별 이메일 발송
    4. 발송 성공/실패 결과 기록
    5. DB send_log에 발송 이력 저장

    sender를 외부에서 주입할 수 있기 때문에
    실제 이메일 발송 없이 테스트할 수도 있다.
    """

    deployment = prepare_deployment(
        digest_date=digest_date,
        keyword=keyword,
        summaries=summaries,
        insight=insight,
    )

    results = []

    for recipient in deployment["recipients"]:
        try:
            # 현재 email_send.py의 send_email은
            # config.EMAIL_TO를 사용하기 때문에,
            # 테스트/구조 검증에서는 sender를 주입해서 사용한다.
            result = sender(
                deployment["subject"],
                deployment["html"],
            )

            # 발송 성공 기록
            log_send(
                digest_id=digest_id,
                channel="email",
                recipient=recipient,
                status="success",
            )

            results.append(
                {
                    "recipient": recipient,
                    "status": "success",
                    "result": result,
                }
            )

        except Exception as exc:
            # 발송 실패 기록
            log_send(
                digest_id=digest_id,
                channel="email",
                recipient=recipient,
                status="failed",
                error=str(exc),
            )

            results.append(
                {
                    "recipient": recipient,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "subject": deployment["subject"],
        "html": deployment["html"],
        "recipients": results,
        "success_count": sum(
            item["status"] == "success"
            for item in results
        ),
        "total_count": len(results),
    }