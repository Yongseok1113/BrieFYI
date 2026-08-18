from make_train_data.entity_extract import extract


def test_enrichment_있으면_그대로_반환():
    article = {"entity": ["NVIDIA", "TSMC"], "category": "기술", "domain": ["반도체"]}
    entities, category, domains = extract(article)
    assert entities == ["NVIDIA", "TSMC"]
    assert category == "기술"
    assert domains == ["반도체"]


def test_enrichment_없으면_시드_별칭_매칭():
    article = {"title": "엔비디아, TSMC와 공급 계약 확대", "description": "", "entity": None}
    entities, category, domains = extract(article)
    assert set(entities) == {"NVIDIA", "TSMC"}
    assert category is None
    assert domains == []


def test_영문_원문도_매칭된다():
    article = {"title": "OpenAI announces new partnership with Microsoft", "description": "", "entity": None}
    entities, _, _ = extract(article)
    assert set(entities) == {"OpenAI", "Microsoft"}


def test_매칭되는_엔티티_없으면_빈_리스트():
    article = {"title": "동네 카페, 신메뉴 출시", "description": "", "entity": None}
    entities, category, domains = extract(article)
    assert entities == []
    assert category is None
    assert domains == []


def test_description도_함께_검사한다():
    article = {"title": "업계 동향", "description": "삼성전자가 신규 라인을 발표했다", "entity": None}
    entities, _, _ = extract(article)
    assert entities == ["Samsung"]
