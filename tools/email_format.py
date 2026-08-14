"""이메일 포맷터 (구현 항목 #5). 구조화된 데이터를 Jinja2 템플릿으로 HTML 렌더링한다."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))


def format_email(digest_date: str, keyword: str, summaries: list[dict], insight: dict) -> str:
    template = _env.get_template("digest_email.html.j2")
    return template.render(
        digest_date=digest_date,
        keyword=keyword,
        summaries=summaries,
        insights=insight.get("insights", []),
        business_implication=insight.get("business_implication", ""),
    )
