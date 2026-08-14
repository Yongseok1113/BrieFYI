import json
from pathlib import Path

from tools.deployment import (
    load_recipients,
    prepare_deployment,
    validate_deployment_data,
)


def test_load_recipients():
    recipients = load_recipients()

    assert len(recipients) == 2
    assert all("@" in recipient for recipient in recipients)


def test_load_test_data():
    with open("test_text.json", "r", encoding="utf-8") as file:
        summaries = json.load(file)

    assert len(summaries) > 0
    assert "title" in summaries[0]
    assert "description" in summaries[0]


def test_validate_deployment_data():
    summaries = [
        {
            "title": "테스트 기사",
            "description": "테스트 내용",
        }
    ]

    insight = {
        "insights": ["테스트 인사이트"],
        "business_implication": "테스트 영향",
    }

    validate_deployment_data(
        summaries,
        insight,
    )


def test_empty_data_is_rejected():
    try:
        validate_deployment_data([], {})
    except ValueError:
        return

    raise AssertionError(
        "빈 배포 데이터가 검증을 통과했습니다."
    )


def test_prepare_deployment():
    with open("test_text.json", "r", encoding="utf-8") as file:
        summaries = json.load(file)

    insight = {
        "insights": [
            "테스트 인사이트"
        ],
        "business_implication": "테스트 사업적 영향",
    }

    result = prepare_deployment(
        digest_date="2026-08-14",
        keyword="AI",
        summaries=summaries,
        insight=insight,
    )

    assert result["subject"] == "[2026-08-14] AI 뉴스&인사이트"
    assert result["html"]
    assert len(result["recipients"]) == 2