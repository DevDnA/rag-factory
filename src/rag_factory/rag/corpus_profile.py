"""CorpusProfile — RAG 인덱스가 어떤 도메인을 담고 있는지의 자기 기술.

rag-factory는 어떤 도메인의 문서가 들어올지 사전에 알 수 없습니다(의료·법률·
RFP·금융 등). 따라서 라우팅·합성 단계에서 "이 corpus는 무엇을 다루는가"라는
컨텍스트가 있어야 도메인 약어·전문어를 정확히 분류할 수 있습니다.

본 모듈은:
1. 인덱스 시 첫 N개 청크를 표본으로 LLM에 요약을 요청해 ``CorpusProfile`` 생성
2. ``corpus_profile.json``에 영속화
3. 서버 시작 시 로드해 ``IntentClassifier`` 등 라우팅 컴포넌트에 주입

생성 실패는 absorb — 빈 profile로 동작(현재 라우팅과 동일).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger("rag.corpus_profile")


# 영문 약어·표준명 추출 패턴. 한국어 corpus에서 LLM이 평가/심사 섹션 위주로
# 키워드를 골라 기술 약어를 누락하는 문제를 보완하기 위한 휴리스틱.
# 영어 corpus는 일반 단어와 구분이 어려워 사용 권장 X (language="ko" 전제).
#
# 경계 처리: ``\b``는 한글이 ``\w``에 포함되어 "NMS에서" 같은 한국어 조사 인접에서
# 작동하지 않음. 그래서 영문/숫자 비인접만 명시하는 lookaround를 사용.
_BNDL = r"(?<![A-Za-z0-9])"   # 좌측 경계: 영문/숫자 비인접
_BNDR = r"(?![A-Za-z0-9])"    # 우측 경계: 영문/숫자 비인접

_ACRONYM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(_BNDL + r"\d+G" + _BNDR),                                  # 5G, 4G, 3G
    re.compile(_BNDL + r"Wi[Ff]i\s*\d+[A-Z]*" + _BNDR),                   # WiFi 7, WiFi 6E
    re.compile(_BNDL + r"IEEE\s+\d+(?:\.\d+)*(?:[a-z]+)?" + _BNDR),       # IEEE 802.1x
    re.compile(_BNDL + r"WPA\s*\d+" + _BNDR),                             # WPA3, WPA 3
    re.compile(_BNDL + r"Sub-?\d+" + _BNDR),                              # Sub-6
    re.compile(_BNDL + r"FR\d+" + _BNDR),                                 # FR1, FR2
    re.compile(_BNDL + r"[A-Z][a-z]{1,3}[A-Z][a-z]?" + _BNDR),            # QoS, IoT, IaaS, IoTHub (camelCase)
    re.compile(_BNDL + r"[A-Z]{2,6}" + _BNDR),                            # NMS, AP, MIMO, ESG
)

# 한국어 일반 명사 stop-list — 도메인 식별성 낮은 일반 단어.
# kiwipiepy NNG/NNP 추출 후 빈도 cutoff 이전에 제거. RFP·법률·의료 등 한국어
# 도메인 corpus에서 보편적으로 노이즈 작용.
_KOREAN_STOPLIST: frozenset[str] = frozenset({
    # 일반 행위·사물
    "사업", "내용", "제공", "경우", "결과", "방법", "정보", "자료",
    "목적", "대상", "사용", "관리", "운영", "수행", "실시", "처리",
    "기능", "활용", "확인", "제출", "지원", "변경", "추가", "구분",
    "포함", "준수", "이용", "참고", "기재", "작성",
    # 시간·단위·범위
    "기간", "시간", "단계", "사항", "항목", "부문", "범위", "정도",
    "단위", "전체", "종류", "유형", "수준",
    # 일반 부사적
    "관련", "필요", "가능", "적합", "동일", "최대", "최소", "이상",
    "이하", "다음",
    # 짧은 의존명사 (이미 길이 2 cutoff에 걸리지만 대비)
    "것", "수", "등", "때", "외", "내", "후", "전", "위", "안",
    # 단위 (2자)
    "원본", "사본", "여부",
})


# 도메인 무관 일반 약어 — 한국어 corpus에서 노이즈. 빈도 cutoff 이전에 제거.
_ACRONYM_STOPLIST: frozenset[str] = frozenset({
    # 컴퓨팅·일반
    "TV", "USB", "PDF", "HTML", "XML", "JSON", "CSV", "URL", "URI",
    "OK", "OR", "AND", "NOT", "IF", "IS", "AT", "IN", "ON", "BY", "TO",
    "AM", "PM", "DC", "AC", "OS", "PC", "NO", "YES",
    "EU", "UN", "US", "UK", "KR", "JP", "CN",
    "ID", "IP", "IT",
    # 신용평가등급 — RFP·금융 도메인에서 흔하지만 도메인 식별성 낮은 노이즈
    "AAA", "AA", "BBB", "BB", "CCC", "CC", "DDD", "DD",
})


def extract_acronyms(
    chunks: list[str],
    *,
    min_frequency: int = 2,
    max_results: int = 20,
) -> list[str]:
    """청크 텍스트에서 영문 약어·표준명을 빈도순으로 추출합니다 — never raises.

    LLM이 한국어 corpus의 기술 약어를 누락하더라도 청크에 실제 등장한 약어를
    정규식으로 직접 잡아 corpus_profile.keywords에 합쳐주는 보완 장치.
    환각 위험 0 (청크 근거).

    Parameters
    ----------
    chunks:
        대상 청크 텍스트 목록. 보통 corpus_profile 생성용 표본 청크.
    min_frequency:
        이 빈도 이상 등장한 약어만 채택. 우발적 OCR 오타·1회성 토큰 제거.
    max_results:
        반환할 최대 약어 수. 빈도 내림차순 → 알파벳 오름차순으로 안정 정렬.

    Returns
    -------
    list[str]
        빈도 desc 정렬된 약어 목록. 각 약어는 청크 등장 표기 그대로.
    """
    if not chunks:
        return []

    counter: Counter[str] = Counter()
    for chunk in chunks:
        text = chunk or ""
        for pattern in _ACRONYM_PATTERNS:
            for match in pattern.finditer(text):
                token = " ".join(match.group(0).split())  # 내부 공백 정규화
                if len(token) < 2 or token in _ACRONYM_STOPLIST:
                    continue
                counter[token] += 1

    qualified = [(tok, freq) for tok, freq in counter.items() if freq >= min_frequency]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    return [tok for tok, _ in qualified[:max_results]]


def extract_korean_nouns(
    chunks: list[str],
    *,
    min_frequency: int = 2,
    max_results: int = 100,
) -> list[str]:
    """한국어 corpus에서 도메인 명사(NNG/NNP)를 빈도순으로 추출 — never raises.

    영문 약어가 잡지 못하는 도메인 명사(예: 차등점수제·수행실적·정량평가·
    참여자격)를 보완. ``kiwipiepy``가 설치돼 있어야 동작하며 미설치 시 빈
    리스트 반환(graceful degradation).

    Parameters
    ----------
    chunks:
        대상 청크 텍스트 목록. 보통 corpus_profile 생성용 전체 청크 풀.
    min_frequency:
        이 빈도 이상 등장한 명사만 채택. 영문 약어보다 cutoff을 높게 두는
        이유는 한국어 일반 명사가 더 자주 등장하기 때문.
    max_results:
        반환할 최대 명사 수. 빈도 desc → 사전 asc로 안정 정렬. 단일 도메인
        corpus는 20개로도 도메인 식별이 충분하지만, 다중 도메인 corpus
        (예: 법령·금융·보도자료 혼합)에서는 20개가 dominant 1~2 문서의
        용어로 편중되어 in-domain gate가 OOD를 잘못 거절하는 land mine을
        만든다. 100개 정도로 늘리면 차순위 도메인 명사도 noise 없이 포함
        되며 (1자 명사·stop-list로 차단), keyword 매칭이 substring 기반
        이므로 합성 비용은 미미하다.

    Returns
    -------
    list[str]
        빈도 desc 정렬된 한국어 도메인 명사 목록. 각 명사는 청크 등장
        표면형 그대로(kiwipiepy의 NNG/NNP만, 1자 명사·stop-list 제외).
    """
    if not chunks:
        return []
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        logger.debug("kiwipiepy 미설치 — extract_korean_nouns 건너뜀")
        return []

    try:
        kiwi = Kiwi()
    except Exception as exc:
        logger.warning("kiwipiepy 초기화 실패: %s", exc)
        return []

    counter: Counter[str] = Counter()
    for chunk in chunks:
        text = chunk or ""
        if not text.strip():
            continue
        try:
            tokens = kiwi.tokenize(text)
        except Exception as exc:
            logger.debug("kiwipiepy tokenize 실패(skip chunk): %s", exc)
            continue
        for tok in tokens:
            tag = getattr(tok, "tag", "")
            if not (tag == "NNG" or tag == "NNP"):
                continue
            form = getattr(tok, "form", "") or ""
            if len(form) < 2 or form in _KOREAN_STOPLIST:
                continue
            counter[form] += 1

    qualified = [(t, f) for t, f in counter.items() if f >= min_frequency]
    qualified.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in qualified[:max_results]]


# name 토큰화에서 제외할 한국어 조사·접속사·기능어. 짧고 의미 약한 토큰이
# IntentClassifier 헤더에 노출되면 매칭 잡음만 늘어남.
_NAME_STOPWORDS: frozenset[str] = frozenset(
    {"및", "또는", "그리고", "관련", "전용", "기반", "위한", "대한", "통한"}
)


def _extract_name_tokens(name: str) -> list[str]:
    """CorpusProfile.name에서 의미 토큰(2자 이상, 비기능어)을 추출합니다.

    예: "소프트웨어 및 사업지원 RFP 입찰 제안서"
        → ["소프트웨어", "사업지원", "RFP", "입찰", "제안서"]
    """
    if not name:
        return []
    parts = re.split(r"[\s,()\[\]<>'\"·•\-/]+", name)
    out: list[str] = []
    for raw in parts:
        token = raw.strip()
        if len(token) < 2:
            continue
        if token in _NAME_STOPWORDS:
            continue
        out.append(token)
    return out


@dataclass(frozen=True)
class CorpusProfile:
    """인덱스 corpus의 도메인 자기 기술.

    Attributes
    ----------
    name:
        한 줄 명칭 — 예: "한국 통신사 RFP 문서". 빈 문자열이면 미설정.
    summary:
        2~5문장 요약. 라우팅·합성 프롬프트 헤더에 주입됩니다.
    keywords:
        도메인 핵심 키워드/약어 — 예: ["NMS", "BIS", "4×4 MIMO"].
        IntentClassifier가 약어 false negative를 줄이는 데 사용.
    generated_at:
        ISO-8601 생성 시각.
    model:
        프로파일을 생성한 LLM 모델명.
    chunks_sampled:
        프로파일 생성에 사용된 표본 청크 수.
    """

    name: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    generated_at: str = ""
    model: str = ""
    chunks_sampled: int = 0

    def is_empty(self) -> bool:
        """name·summary·keywords가 모두 비어 있으면 True (라우팅 컨텍스트 미주입)."""
        return not self.name and not self.summary and not self.keywords

    def merged_keywords(self) -> list[str]:
        """name의 의미 토큰을 keywords와 합쳐 중복 제거한 리스트.

        IntentClassifier가 `name`에 포함된 약어·표제어(예: "RFP")를 keywords와
        동등하게 매칭하도록 하기 위함. 매칭은 corpus_override 보조 신호로도 사용.
        """
        tokens = _extract_name_tokens(self.name)
        seen: set[str] = set()
        out: list[str] = []
        for k in list(self.keywords) + tokens:
            if not k:
                continue
            key = k.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(k)
        return out

    def to_prompt_header(self) -> str:
        """IntentClassifier·합성 프롬프트에 주입할 헤더 텍스트.

        빈 profile이면 빈 문자열을 반환합니다 — 호출 측이 그대로 concat하면 됩니다.
        """
        if self.is_empty():
            return ""
        lines: list[str] = ["[본 corpus 도메인 정보]"]
        if self.name:
            lines.append(f"- 명칭: {self.name}")
        if self.summary:
            lines.append(f"- 요약: {self.summary}")
        merged = self.merged_keywords()
        if merged:
            # cap 60 — name 토큰까지 합쳐 약간 늘림. IntentClassifier 비용 미미.
            lines.append(f"- 핵심 키워드: {', '.join(merged[:60])}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 영속화
# ---------------------------------------------------------------------------


def load_corpus_profile(path: Path) -> CorpusProfile:
    """JSON 파일에서 CorpusProfile을 로드합니다 — never raises.

    파일이 없거나 파싱 실패 시 빈 profile을 반환합니다.
    """
    if not path.exists():
        return CorpusProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("CorpusProfile 로드 실패 (%s): %s — 빈 profile 사용", path, exc)
        return CorpusProfile()

    keywords = data.get("keywords", []) or []
    if not isinstance(keywords, list):
        keywords = []

    return CorpusProfile(
        name=str(data.get("name", "") or ""),
        summary=str(data.get("summary", "") or ""),
        keywords=[str(k) for k in keywords if k],
        generated_at=str(data.get("generated_at", "") or ""),
        model=str(data.get("model", "") or ""),
        chunks_sampled=int(data.get("chunks_sampled", 0) or 0),
    )


def save_corpus_profile(profile: CorpusProfile, path: Path) -> None:
    """CorpusProfile을 JSON 파일로 영속화합니다 — never raises.

    저장 실패는 로그만 남기고 호출 측에 전파하지 않습니다(인덱싱·서버 가용성 우선).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(profile.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("CorpusProfile 저장 실패 (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# 자동 생성기
# ---------------------------------------------------------------------------


_PROFILE_PROMPT = """다음은 어떤 RAG 인덱스에서 추출한 표본 청크 {n}개입니다.
이 corpus가 어떤 도메인을 다루는지 분석하여 JSON으로 답하세요.

[표본 청크]
{samples}

규칙:
- "name": **이 corpus의 고유 사업·문서를 식별하는 한 줄 명명**. RFP·계약·사양서라면 사업명·시스템명을
  명시해야 함. 표본 중 일부에 일반 평가 기준·양식·법적 boilerplate가 섞여 있어도 그것을 name으로
  채택하지 마세요. 청크 중 **고유성 높은 1-2개**(특정 사업명·제품명·서비스명을 명시하는 청크)가
  도메인의 정체성을 결정합니다.
  - 좋은 예: "버스 공공와이파이 임차운영 사업 RFP", "OO 시 도시철도 신호제어 시스템 구축 제안서"
  - 나쁜 예: "소프트웨어 및 사업지원 RFP 입찰 제안서", "공공조달 평가 기준 안내서" — 너무 generic.
    이런 명칭은 거의 모든 공공 RFP corpus와 구분되지 않으므로 IntentClassifier가 in-domain 질의를
    out-of-domain으로 오분류하게 만듭니다.
- "summary": 2~4문장으로 corpus가 다루는 주제·범위를 요약. **이 사업의 고유 대상·규모·기술 스택**을
  포함하세요(예: 대수·구간·기술 키워드). "평가 기준"·"수행실적"·"신용평가"는 거의 모든 RFP에 등장하므로
  summary의 핵심 정체성으로 사용하지 마세요 — 부속·보조 정보입니다.
- "keywords": 도메인 식별 키워드 **15~25개**. 다음 카테고리를 골고루 포함하세요:
    1) **기술 약어·표준명**: 청크에 등장하는 기술 약어, 규격, 프로토콜 — 예: 5G, WiFi 7, NMS, AP, IEEE 802.1x, WPA3, MIMO, FR1, Sub-6
    2) **고유 명사·기관명**: 사업명, 기관명, 시스템명, 제품명 — 예: 한국지능정보사회진흥원, BIS, 공공와이파이 통합관리센터
    3) **도메인 핵심 개념**: 그 도메인 사람만 자주 쓰는 명사 — 예: 백홀, 대개체, 차등점수제, SLA, 정량평가, 정성평가
    4) **수치 단위·기준명**: 평가/품질/기준 명칭 — 예: 수행실적, 신용평가등급, 배점한도
  ⚠ "버스", "사업", "운영" 같은 일반 명사보다 위 4개 카테고리에 해당하는 **구체적·식별성 있는** 단어를 우선.
  ⚠ 영문 약어는 청크에 등장한 표기 그대로 (예: "WiFi 7", "5G", "WPA 3"). 한글로 음역하지 말 것.
- 모든 텍스트 필드(name, summary)는 한국어로 작성. keywords는 청크 표기 그대로.
- 청크에 없는 정보는 추측하지 마세요. 그러나 표본 청크 중 일부에 **사업명·제품명·서비스명을 명시하는
  고유성 높은 정보**가 있으면 그것이 corpus의 핵심 정체성을 대표합니다.

반드시 다음 JSON 형식으로만 답변하세요 (다른 텍스트 금지):
{{
  "name": "...",
  "summary": "...",
  "keywords": ["...", "...", "..."]
}}

## 예시

[표본 청크]
[청크 1]
o 5G 백홀 기반 WiFi 7 AP를 32,857대 대개체. AP는 3개 이상 제조사 구성원 필요.
   IEEE 802.1x 인증, WPA 3 Enterprise/OWE 보안 적용. NMS에서 원격 제어.
[청크 2]
다. 제안서 평가 기준 및 배점(100점). 정량평가(10점): 경영상태, 수행실적, AP 제조사 수.
   정성평가(90점): 전략 및 방법론, 안전관리, 보안관리, ESG.

{{
  "name": "버스 공공와이파이 임차운영 사업 RFP 제안 안내서",
  "summary": "32,857대 시내버스에 5G 백홀 WiFi 7 AP를 대개체·운영하는 임차사업 입찰 제안 안내서. AP 기술 사양, NMS 관리 체계, SLA 기준, 평가 배점(정량/정성)과 차등점수제 등을 다룬다.",
  "keywords": ["5G", "5G 백홀", "WiFi 7", "AP", "NMS", "BIS", "IEEE 802.1x", "WPA 3", "OWE", "MIMO", "FR1", "Sub-6", "Preamble Puncturing", "대개체", "백홀", "정량평가", "정성평가", "차등점수제", "수행실적", "경영상태", "신용평가등급", "SLA", "장애 복구", "한국지능정보사회진흥원", "공공와이파이 통합관리센터"]
}}
"""


def _format_samples(chunks: list[str], max_chars_per_chunk: int = 800) -> str:
    """청크 목록을 LLM 프롬프트에 넣을 텍스트로 포맷합니다."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        parts.append(f"[청크 {i}]\n{text}")
    return "\n\n".join(parts)


def _parse_profile_json(raw: str) -> dict | None:
    """LLM 응답에서 JSON 객체를 추출합니다 — 실패 시 None."""
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return None
    try:
        parsed = json.loads(cleaned[brace_start : brace_end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def generate_corpus_profile(
    *,
    chunks: list[str],
    http_client: Any,
    ollama_model: str,
    api_base: str,
    request_timeout: float = 60.0,
    keep_alive: str = "5m",
    max_tokens: int = 1200,
    acronym_chunks: list[str] | None = None,
) -> CorpusProfile:
    """LLM으로 CorpusProfile을 자동 생성합니다 — never raises.

    Parameters
    ----------
    chunks:
        표본 청크 텍스트 목록. 권장 8~16개.
    http_client:
        Ollama ``/api/generate``를 호출할 ``httpx.AsyncClient``.
    ollama_model:
        프로파일 생성에 사용할 모델명.
    api_base:
        Ollama API 베이스 URL.

    Returns
    -------
    CorpusProfile
        생성된 profile. 실패 시 빈 profile.
    """
    from datetime import datetime, timezone

    if not chunks:
        return CorpusProfile()

    prompt = _PROFILE_PROMPT.format(
        n=len(chunks),
        samples=_format_samples(chunks),
    )

    try:
        response = await http_client.post(
            f"{api_base}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": "json",
                "keep_alive": keep_alive,
                "options": {
                    "num_predict": max_tokens,
                    # Phase 14 — profile 생성은 인덱싱 시 1회만 실행되므로 결정성이
                    # 더 중요. default sampling에서는 dominant theme 추정이 흔들려
                    # 한 번 잘못 저장되면 영구히 잘못된 routing이 고착. temperature=0
                    # + seed로 동일 corpus → 동일 profile 보장.
                    "temperature": 0,
                    "seed": 42,
                    "top_p": 1.0,
                    "top_k": 1,
                },
            },
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        raw = data.get("response", "") or data.get("thinking", "")
    except Exception as exc:
        logger.warning("CorpusProfile LLM 호출 실패: %s — 빈 profile 사용", exc)
        return CorpusProfile()

    parsed = _parse_profile_json(raw)
    if parsed is None:
        logger.warning("CorpusProfile JSON 파싱 실패 — 빈 profile 사용")
        return CorpusProfile()

    raw_keywords = parsed.get("keywords", []) or []
    if not isinstance(raw_keywords, list):
        raw_keywords = []
    keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]

    # 휴리스틱 키워드 추출로 LLM 누락 보완. ``acronym_chunks``가 주어지면 전체
    # 코퍼스 풀에서 추출(LLM 표본은 작아도 빈도 cutoff을 통과시키기 위함). 미주어
    # 지면 LLM 표본 청크로 추출 — 하위 호환.
    #
    # 두 가지 추출:
    # 1) 영문 약어·표준명 — IntentClassifier가 도메인 약어를 정확히 인식
    # 2) 한국어 도메인 명사 — kiwipiepy NNG/NNP, 영문이 못 잡는 한국어 도메인 용어
    #    (차등점수제·수행실적·정량평가 등) 보강
    #
    # 순서는 영문 → 한국어 → LLM. 도메인 식별성이 높은 순으로 prompt header cap에
    # 안정적으로 노출되도록.
    pool = acronym_chunks if acronym_chunks else chunks
    acronyms = extract_acronyms(pool)
    korean_nouns = extract_korean_nouns(pool)
    seen_lower = {a.lower() for a in acronyms}
    deduped_korean: list[str] = []
    for n in korean_nouns:
        low = n.lower()
        if low in seen_lower:
            continue
        deduped_korean.append(n)
        seen_lower.add(low)
    deduped_llm = [k for k in keywords if k.lower() not in seen_lower]
    keywords = acronyms + deduped_korean + deduped_llm

    return CorpusProfile(
        name=str(parsed.get("name", "") or "").strip(),
        summary=str(parsed.get("summary", "") or "").strip(),
        keywords=keywords,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=ollama_model,
        chunks_sampled=len(chunks),
    )


def merge_with_override(
    auto: CorpusProfile,
    *,
    name_override: str = "",
    summary_override: str = "",
    keywords_override: list[str] | None = None,
) -> CorpusProfile:
    """자동 생성 profile에 사용자 override를 적용합니다.

    각 필드별로 override가 비어 있지 않으면 우선시합니다. keywords는 override가
    제공되면 완전 대체(append 아님) — 사용자가 의도한 키워드만 노출되도록.
    """
    keywords = (
        list(keywords_override)
        if keywords_override is not None and len(keywords_override) > 0
        else list(auto.keywords)
    )
    return CorpusProfile(
        name=name_override.strip() or auto.name,
        summary=summary_override.strip() or auto.summary,
        keywords=keywords,
        generated_at=auto.generated_at,
        model=auto.model,
        chunks_sampled=auto.chunks_sampled,
    )


__all__ = [
    "CorpusProfile",
    "extract_acronyms",
    "extract_korean_nouns",
    "generate_corpus_profile",
    "load_corpus_profile",
    "merge_with_override",
    "save_corpus_profile",
]
