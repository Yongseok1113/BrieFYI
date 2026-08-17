"""data_pipeline: 프롬프트 엔지니어링 기반 학습 데이터 생성 파이프라인.

docs/data-pipeline-design.md 참고. 레포 루트의 db/, tools/ 등을 import하지 않는
자체 완결(self-contained) 패키지다 — 별도 Docker 이미지로 빌드되기 때문.
"""

__version__ = "0.1.0"
