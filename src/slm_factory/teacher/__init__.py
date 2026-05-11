"""교사 LLM 추상화 계층.

다양한 백엔드(Ollama, OpenAI 호환 API 등)를 통해 대규모 교사 모델에서 텍스트를 생성하기 위한
통일된 인터페이스를 제공합니다.
"""

from __future__ import annotations

from ..config import TeacherConfig
from .base import BaseTeacher
from .ollama import OllamaTeacher
from .openai_compat import OpenAICompatTeacher

__all__ = [
    "BaseTeacher",
    "OllamaTeacher",
    "OpenAICompatTeacher",
    "create_teacher",
]


def create_teacher(config: TeacherConfig) -> BaseTeacher:
    """*config*에서 적절한 교사 백엔드를 인스턴스화합니다.

    매개변수
    ----------
    config:
        ``backend`` 필드가 구현을 선택하는 :class:`TeacherConfig`
        (``"ollama"`` 또는 ``"openai"``).

    반환값
    -------
    BaseTeacher
        사용 가능한 교사 인스턴스.

    예외
    ------
    ValueError
        ``config.backend``가 인식되지 않는 경우.
    """
    if config.backend == "ollama":
        return OllamaTeacher(config)
    elif config.backend == "openai":
        return OpenAICompatTeacher(config)
    else:
        raise ValueError(
            f"Unknown teacher backend: {config.backend!r}. "
            f"Supported backends: 'ollama', 'openai'."
        )
