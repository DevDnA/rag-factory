"""유틸리티(utils.py)의 단위 테스트입니다."""

import asyncio
import hashlib
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import slm_factory.utils as utils_module
from slm_factory.utils import compute_file_hash, get_logger, run_bounded, setup_logging


class TestSetupLogging:
    """setup_logging 함수의 테스트입니다."""

    def test_로거_반환(self):
        """setup_logging이 logging.Logger 인스턴스를 반환하는지 확인합니다."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "slm_factory"

    def test_중복_호출_시_basicConfig_한번만_호출(self, mocker):
        """setup_logging을 여러 번 호출해도 basicConfig가 한 번만 호출되는지 확인합니다."""
        # _configured를 False로 리셋하여 테스트 시작
        utils_module._configured = False
        mock_basic = mocker.patch("logging.basicConfig")

        setup_logging()
        setup_logging()
        setup_logging()

        # basicConfig는 최초 한 번만 호출되어야 합니다
        mock_basic.assert_called_once()

        # 테스트 후 상태 복원
        utils_module._configured = True


class TestGetLogger:
    """get_logger 함수의 테스트입니다."""

    def test_올바른_이름의_로거_반환(self):
        """get_logger가 'slm_factory.{name}' 형식의 로거를 반환하는지 확인합니다."""
        logger = get_logger("parsers.text")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "slm_factory.parsers.text"

    def test_다른_이름으로_다른_로거_반환(self):
        """서로 다른 이름으로 호출하면 서로 다른 로거를 반환하는지 확인합니다."""
        logger_a = get_logger("module_a")
        logger_b = get_logger("module_b")
        assert logger_a.name != logger_b.name
        assert logger_a.name == "slm_factory.module_a"
        assert logger_b.name == "slm_factory.module_b"


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    """compute_file_hash 함수의 테스트입니다."""

    def test_sha256_기본_해시(self, tmp_path: Path):
        """기본 알고리즘(sha256)으로 올바른 해시를 반환하는지 확인합니다."""
        f = tmp_path / "sample.txt"
        content = b"hello world"
        f.write_bytes(content)

        result = compute_file_hash(f)
        expected = hashlib.sha256(content).hexdigest()

        assert result == expected

    def test_md5_알고리즘(self, tmp_path: Path):
        """md5 알고리즘을 지정하면 올바른 해시를 반환하는지 확인합니다."""
        f = tmp_path / "sample.txt"
        content = b"test data for md5"
        f.write_bytes(content)

        result = compute_file_hash(f, algorithm="md5")
        expected = hashlib.md5(content).hexdigest()

        assert result == expected

    def test_빈_파일(self, tmp_path: Path):
        """빈 파일의 해시가 올바르게 계산되는지 확인합니다."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")

        result = compute_file_hash(f)
        expected = hashlib.sha256(b"").hexdigest()

        assert result == expected

    def test_동일_내용_동일_해시(self, tmp_path: Path):
        """동일 내용의 파일은 동일한 해시를 반환하는지 확인합니다."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")

        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_다른_내용_다른_해시(self, tmp_path: Path):
        """다른 내용의 파일은 다른 해시를 반환하는지 확인합니다."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_str_경로_허용(self, tmp_path: Path):
        """문자열 경로도 허용되는지 확인합니다."""
        f = tmp_path / "sample.txt"
        f.write_bytes(b"string path test")

        result_path = compute_file_hash(f)
        result_str = compute_file_hash(str(f))

        assert result_path == result_str

    def test_존재하지_않는_파일(self, tmp_path: Path):
        """존재하지 않는 파일이면 FileNotFoundError가 발생하는지 확인합니다."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash(tmp_path / "nonexistent.txt")


# ---------------------------------------------------------------------------
# run_bounded
# ---------------------------------------------------------------------------


class TestRunBounded:
    """run_bounded 함수의 테스트입니다."""

    def test_반환값_전파(self):
        """코루틴의 반환값이 그대로 전파되는지 확인합니다."""

        async def _coro():
            return 42

        progress = MagicMock()
        sem = asyncio.Semaphore(1)

        result = asyncio.run(run_bounded(sem, _coro(), progress, task_id=0))

        assert result == 42

    def test_진행률_갱신(self):
        """실행 후 progress.advance가 호출되는지 확인합니다."""

        async def _coro():
            return "ok"

        progress = MagicMock()
        sem = asyncio.Semaphore(1)

        asyncio.run(run_bounded(sem, _coro(), progress, task_id=7))

        progress.advance.assert_called_once_with(7)

    def test_세마포어_동시성_제한(self):
        """세마포어가 동시 실행 수를 제한하는지 확인합니다."""
        max_concurrent = 0
        current = 0

        async def _track():
            nonlocal max_concurrent, current
            current += 1
            if current > max_concurrent:
                max_concurrent = current
            await asyncio.sleep(0.01)
            current -= 1
            return True

        async def _run():
            sem = asyncio.Semaphore(2)
            progress = MagicMock()
            tasks = [run_bounded(sem, _track(), progress, task_id=0) for _ in range(6)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())

        assert all(results)
        assert max_concurrent <= 2

    def test_예외_전파(self):
        """코루틴이 예외를 발생시키면 그대로 전파되는지 확인합니다."""

        async def _fail():
            raise ValueError("의도된 오류")

        progress = MagicMock()
        sem = asyncio.Semaphore(1)

        with pytest.raises(ValueError, match="의도된 오류"):
            asyncio.run(run_bounded(sem, _fail(), progress, task_id=0))
