"""
노트북 검증 유틸리티

사용법 (노트북에서):
    import sys; sys.path.insert(0, '..')
    from common.test_helpers import show_graph, run_and_trace
"""

from IPython.display import display, Image, Markdown


def show_graph(compiled_graph):
    """컴파일된 그래프를 시각화합니다."""
    try:
        png = compiled_graph.get_graph().draw_mermaid_png()
        display(Image(png))
    except Exception:
        mermaid = compiled_graph.get_graph().draw_mermaid()
        display(Markdown(f"```mermaid\n{mermaid}\n```"))


def run_and_trace(compiled_graph, initial_state: dict) -> list:
    """그래프를 실행하고 각 노드의 출력을 추적합니다."""
    print("=" * 60)
    events = []
    for event in compiled_graph.stream(initial_state):
        events.append(event)
        node_name = list(event.keys())[0]
        print(f"\n>> 노드 실행: {node_name}")
        for key, val in event[node_name].items():
            print(f"   {key}: {_truncate(str(val))}")
        print("-" * 40)
    print("=" * 60)
    print("실행 완료!")
    return events


def show_state_diff(before: dict, node_output: dict, node_name: str = ""):
    """노드 실행 전후의 상태 변경을 시각적으로 출력합니다."""
    header = f"State Diff: {node_name}" if node_name else "State Diff"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    changes_found = False
    for key, new_value in node_output.items():
        old_value = before.get(key)
        if old_value != new_value:
            changes_found = True
            print(f"\n  [변경] {key}:")
            print(f"    이전: {_truncate(str(old_value))}")
            print(f"    이후: {_truncate(str(new_value))}")

    if not changes_found:
        print("\n  (변경사항 없음)")
    print(f"\n{'=' * 60}\n")


def _truncate(text: str, max_len: int = 80) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text
