"""환경 검증 스크립트 - 실습에 필요한 패키지가 설치되었는지 확인합니다."""

import sys

REQUIRED_PACKAGES = [
    ("langgraph", "langgraph"),
    ("langchain_core", "langchain-core"),
    ("langchain_anthropic", "langchain-anthropic"),
    ("pydantic", "pydantic"),
    ("IPython", "jupyter"),
    ("httpx", "httpx"),
    ("tenacity", "tenacity"),
    ("tiktoken", "tiktoken"),
    ("yaml", "pyyaml"),
    ("dotenv", "python-dotenv"),
]

OPTIONAL_PACKAGES = [
    ("deepdiff", "deepdiff"),
]


def check_packages():
    print("=" * 50)
    print("  LangGraph 실습 환경 검증")
    print("=" * 50)
    print(f"\nPython 버전: {sys.version}\n")

    all_ok = True

    print("[필수 패키지]")
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {pip_name:25s} ({version})")
        except ImportError:
            print(f"  ❌ {pip_name:25s} (미설치 - pip install {pip_name})")
            all_ok = False

    print("\n[선택 패키지]")
    for import_name, pip_name in OPTIONAL_PACKAGES:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {pip_name:25s} ({version})")
        except ImportError:
            print(f"  ⚠️  {pip_name:25s} (미설치 - 없어도 대부분 실습 가능)")

    print("\n[API 키]")
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        print(f"  ✅ ANTHROPIC_API_KEY 설정됨 ({api_key[:8]}...)")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY 미설정 (Module 04부터 실제 API 호출 시 필요)")

    print("\n" + "=" * 50)
    if all_ok:
        print("  모든 필수 패키지가 설치되어 있습니다!")
        print("  실습을 시작할 준비가 되었습니다.")
    else:
        print("  일부 필수 패키지가 누락되었습니다.")
        print("  pip install -r requirements.txt 를 실행하세요.")
    print("=" * 50)

    return all_ok


if __name__ == "__main__":
    success = check_packages()
    sys.exit(0 if success else 1)
