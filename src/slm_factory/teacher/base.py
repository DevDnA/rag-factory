"""교사 LLM 백엔드를 위한 추상 기본 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTeacher(ABC):
    """모든 교사 백엔드가 구현해야 하는 인터페이스."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """*prompt*를 교사 LLM에 전송하고 응답 텍스트를 반환합니다.

        매개변수
        ----------
        prompt:
            전체 프롬프트 문자열(시스템 + 컨텍스트 + 질문이 이미 병합됨).
        **kwargs:
            백엔드별 오버라이드(예: ``temperature``, ``format``).

        반환값
        -------
        str
            교사 모델에서 생성된 텍스트.
        """
        ...

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        """:meth:`generate`의 비동기 변형입니다.

        기본 구현은 동기식 ``generate()`` 호출을 래핑합니다.
        서브클래스는 동시성을 위해 진정한 비동기 I/O로 오버라이드해야 합니다.

        매개변수
        ----------
        prompt:
            전체 프롬프트 문자열.
        **kwargs:
            백엔드별 오버라이드.

        반환값
        -------
        str
            교사 모델에서 생성된 텍스트.
        """
        return self.generate(prompt, **kwargs)

    def health_check(self) -> bool:
        """백엔드에 도달할 수 있는지 확인합니다.

        백엔드가 성공적으로 응답하면 ``True``를 반환하고,
        그렇지 않으면 ``False``를 반환합니다. 구현은 절대 예외를 발생시키지 않아야 합니다.
        """
        return True
