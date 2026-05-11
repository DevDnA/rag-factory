"""QA 쌍 수동 리뷰 TUI 앱입니다."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static, TextArea

from ..models import QAPair
from ..utils import get_logger
from .widgets import QACard, StatusBar

logger = get_logger("reviewer")


# ---------------------------------------------------------------------------
# 편집 화면
# ---------------------------------------------------------------------------


class EditScreen(Screen[str | None]):
    """QA 답변을 수정하는 편집 화면입니다."""

    BINDINGS = [
        Binding("escape", "cancel", "취소"),
    ]

    DEFAULT_CSS = """
    EditScreen {
        align: center middle;
    }
    #edit-container {
        width: 80%;
        height: 80%;
        border: solid $accent;
        padding: 1 2;
    }
    #edit-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #edit-area {
        height: 1fr;
        margin-bottom: 1;
    }
    #edit-buttons {
        height: 3;
        align: center middle;
    }
    #edit-buttons Button {
        margin: 0 2;
    }
    """

    def __init__(self, answer: str) -> None:
        """편집 화면을 초기화합니다.

        매개변수
        ----------
        answer:
            현재 답변 텍스트입니다.
        """
        super().__init__()
        self._answer = answer

    def compose(self) -> ComposeResult:
        """편집 화면 위젯을 구성합니다."""
        with Static(id="edit-container"):
            yield Static("[bold]✏️ 답변 수정[/bold]", id="edit-title")
            yield TextArea(self._answer, id="edit-area")
            with Static(id="edit-buttons"):
                yield Button("확인", variant="success", id="edit-ok")
                yield Button("취소", variant="error", id="edit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """버튼 클릭 이벤트를 처리합니다."""
        if event.button.id == "edit-ok":
            area = self.query_one("#edit-area", TextArea)
            self.dismiss(area.text)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """편집을 취소합니다."""
        self.dismiss(None)


# ---------------------------------------------------------------------------
# 메인 리뷰 앱
# ---------------------------------------------------------------------------


class QAReviewerApp(App[None]):
    """QA 쌍을 수동으로 리뷰하는 TUI 앱입니다."""

    TITLE = "SLM Factory — QA 리뷰어"

    BINDINGS = [
        Binding("a", "approve", "승인"),
        Binding("r", "reject", "거부"),
        Binding("e", "edit", "수정"),
        Binding("n", "next", "다음"),
        Binding("right", "next", "다음", show=False),
        Binding("p", "prev", "이전"),
        Binding("left", "prev", "이전", show=False),
        Binding("s", "save", "저장"),
        Binding("q", "quit_app", "종료"),
    ]

    CSS = """
    Screen {
        background: $surface;
    }
    #qa-card {
        height: 1fr;
    }
    #status-bar {
        dock: bottom;
        height: 1;
    }
    """

    def __init__(
        self,
        pairs: list[QAPair],
        output_path: Path,
    ) -> None:
        """리뷰 앱을 초기화합니다.

        매개변수
        ----------
        pairs:
            리뷰할 QA 쌍 목록입니다.
        output_path:
            리뷰 결과를 저장할 파일 경로입니다.
        """
        super().__init__()
        self.pairs: list[QAPair] = pairs
        self.current_index: int = 0
        self.unsaved: bool = False
        self.output_path: Path = output_path

    # ------------------------------------------------------------------
    # 화면 구성
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """앱 위젯을 구성합니다."""
        yield Header()
        yield QACard(id="qa-card")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """앱 마운트 시 첫 번째 QA 쌍을 표시합니다."""
        self._refresh_display()

    # ------------------------------------------------------------------
    # 액션
    # ------------------------------------------------------------------

    def action_approve(self) -> None:
        """현재 QA 쌍을 승인하고 다음으로 이동합니다."""
        if not self.pairs:
            return
        self.pairs[self.current_index].review_status = "approved"
        self.unsaved = True
        self._advance()

    def action_reject(self) -> None:
        """현재 QA 쌍을 거부하고 다음으로 이동합니다."""
        if not self.pairs:
            return
        self.pairs[self.current_index].review_status = "rejected"
        self.unsaved = True
        self._advance()

    def action_next(self) -> None:
        """다음 QA 쌍으로 이동합니다."""
        if not self.pairs:
            return
        if self.current_index < len(self.pairs) - 1:
            self.current_index += 1
            self._refresh_display()

    def action_prev(self) -> None:
        """이전 QA 쌍으로 이동합니다."""
        if not self.pairs:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self._refresh_display()

    def action_save(self) -> None:
        """리뷰 결과를 JSON 파일로 저장합니다."""
        self.save_pairs(self.pairs, self.output_path)
        self.unsaved = False
        self.notify("저장 완료!", severity="information")
        logger.info("리뷰 결과 저장: %s (%d건)", self.output_path, len(self.pairs))

    def action_edit(self) -> None:
        """현재 QA 쌍의 답변을 편집합니다."""
        if not self.pairs:
            return
        pair = self.pairs[self.current_index]
        self.push_screen(EditScreen(pair.answer), callback=self._on_edit_result)

    def _on_edit_result(self, result: str | None) -> None:
        """편집 결과를 반영합니다."""
        if result is not None:
            self.pairs[self.current_index].answer = result
            self.unsaved = True
            self._refresh_display()

    def action_quit_app(self) -> None:
        """앱을 종료합니다. 미저장 변경이 있으면 알림을 표시합니다."""
        if self.unsaved:
            self.notify("미저장 변경이 있습니다. 's'로 저장하세요.", severity="warning")
            return
        self.exit()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """다음 QA 쌍으로 이동하거나 화면을 갱신합니다."""
        if self.current_index < len(self.pairs) - 1:
            self.current_index += 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        """QACard와 StatusBar를 현재 상태로 갱신합니다."""
        if not self.pairs:
            return
        card = self.query_one("#qa-card", QACard)
        card.update_pair(self.pairs[self.current_index])

        statuses = self.count_statuses(self.pairs)
        bar = self.query_one("#status-bar", StatusBar)
        bar.update_status(
            current=self.current_index + 1,
            total=len(self.pairs),
            approved=statuses["approved"],
            rejected=statuses["rejected"],
            pending=statuses["pending"],
        )

    # ------------------------------------------------------------------
    # 정적/클래스 메서드 (테스트 용이성)
    # ------------------------------------------------------------------

    @staticmethod
    def save_pairs(pairs: list[QAPair], path: Path) -> None:
        """QA 쌍 목록을 JSON 파일로 직렬화합니다.

        매개변수
        ----------
        pairs:
            저장할 QA 쌍 목록입니다.
        path:
            저장 경로입니다.
        """
        data = [asdict(p) for p in pairs]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load_pairs(path: Path) -> list[QAPair]:
        """JSON 파일에서 QA 쌍 목록을 로드합니다.

        매개변수
        ----------
        path:
            QA 데이터 JSON 파일 경로입니다.

        반환값
        -------
        list[QAPair]
            로드된 QA 쌍 목록입니다.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return [QAPair(**item) for item in data]

    @staticmethod
    def count_statuses(pairs: list[QAPair]) -> dict[str, int]:
        """QA 쌍 목록의 리뷰 상태별 건수를 반환합니다.

        매개변수
        ----------
        pairs:
            QA 쌍 목록입니다.

        반환값
        -------
        dict[str, int]
            ``{"approved": n, "rejected": n, "pending": n}`` 형태의 딕셔너리입니다.
        """
        counts = {"approved": 0, "rejected": 0, "pending": 0}
        for pair in pairs:
            if pair.review_status in ("approved", "rejected"):
                counts[pair.review_status] += 1
            else:
                counts["pending"] += 1
        return counts
