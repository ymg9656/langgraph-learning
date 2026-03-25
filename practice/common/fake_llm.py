"""
FakeLLM - API 키 없이 LangGraph 실습을 위한 모의 LLM

사용법 (노트북에서):
    import sys; sys.path.insert(0, '..')
    from common.fake_llm import FakeLLM

    llm = FakeLLM(responses={"분석": "분석 결과입니다.", "요약": "요약입니다."})
    result = llm.invoke("이 문서를 분석해주세요")
    print(result.content)  # "분석 결과입니다."
"""

from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeLLM(BaseChatModel):
    """패턴 매칭 기반 가짜 LLM.

    미리 정의한 패턴-응답 매핑에 따라 응답을 반환합니다.
    실제 LLM API를 호출하지 않으므로 비용이 발생하지 않습니다.

    Args:
        responses: 패턴(키) → 응답(값) 딕셔너리. 메시지에 패턴이 포함되면 해당 응답 반환
        default_response: 매칭되는 패턴이 없을 때 반환할 기본 응답
    """

    responses: dict[str, str] = {}
    default_response: str = "FakeLLM 기본 응답입니다."

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "fake-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_message = messages[-1].content if messages else ""

        matched_response = self.default_response
        for pattern, response in self.responses.items():
            if pattern.lower() in last_message.lower():
                matched_response = response
                break

        message = AIMessage(content=matched_response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
