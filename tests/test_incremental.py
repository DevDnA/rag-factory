"""증분 학습 추적기(incremental.py) 모듈의 단위 테스트입니다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_factory.incremental import IncrementalTracker


# ---------------------------------------------------------------------------
# compute_document_hashes
# ---------------------------------------------------------------------------


class TestComputeDocumentHashes:

    def test_빈_디렉토리(self, make_config, tmp_path):
        tracker = IncrementalTracker(make_config())
        result = tracker.compute_document_hashes(tmp_path, ["txt", "pdf"])

        assert result == {}

    def test_존재하지_않는_디렉토리(self, make_config, tmp_path):
        tracker = IncrementalTracker(make_config())
        result = tracker.compute_document_hashes(tmp_path / "nope", ["txt"])

        assert result == {}

    def test_지원_형식_파일만_해시(self, make_config, tmp_path):
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-fake")
        (tmp_path / "c.jpg").write_bytes(b"\xff\xd8")

        tracker = IncrementalTracker(make_config())
        result = tracker.compute_document_hashes(tmp_path, ["txt", "pdf"])

        assert "a.txt" in result
        assert "b.pdf" in result
        assert "c.jpg" not in result

    def test_해시_일관성(self, make_config, tmp_path):
        (tmp_path / "doc.txt").write_text("consistent content", encoding="utf-8")

        tracker = IncrementalTracker(make_config())
        h1 = tracker.compute_document_hashes(tmp_path, ["txt"])
        h2 = tracker.compute_document_hashes(tmp_path, ["txt"])

        assert h1 == h2

    def test_점_접두사_형식_처리(self, make_config, tmp_path):
        (tmp_path / "doc.txt").write_text("data", encoding="utf-8")

        tracker = IncrementalTracker(make_config())
        result = tracker.compute_document_hashes(tmp_path, [".txt"])

        assert "doc.txt" in result


# ---------------------------------------------------------------------------
# load_saved_hashes / save_hashes
# ---------------------------------------------------------------------------


class TestLoadSavedHashes:

    def test_파일_없으면_빈_딕셔너리(self, make_config, tmp_path):
        tracker = IncrementalTracker(make_config())
        result = tracker.load_saved_hashes(tmp_path / "missing.json")

        assert result == {}

    def test_기존_파일_로드(self, make_config, tmp_path):
        hash_file = tmp_path / "hashes.json"
        expected = {"a.txt": "abc123", "b.pdf": "def456"}
        hash_file.write_text(json.dumps(expected), encoding="utf-8")

        tracker = IncrementalTracker(make_config())
        result = tracker.load_saved_hashes(hash_file)

        assert result == expected


class TestSaveHashes:

    def test_유효한_JSON_저장(self, make_config, tmp_path):
        hash_file = tmp_path / "sub" / "hashes.json"
        hashes = {"doc.txt": "abcdef1234567890"}

        tracker = IncrementalTracker(make_config())
        tracker.save_hashes(hashes, hash_file)

        assert hash_file.is_file()
        loaded = json.loads(hash_file.read_text(encoding="utf-8"))
        assert loaded == hashes

    def test_저장_후_로드_일치(self, make_config, tmp_path):
        hash_file = tmp_path / "hashes.json"
        hashes = {"x.txt": "aaa", "y.pdf": "bbb"}

        tracker = IncrementalTracker(make_config())
        tracker.save_hashes(hashes, hash_file)
        loaded = tracker.load_saved_hashes(hash_file)

        assert loaded == hashes


# ---------------------------------------------------------------------------
# detect_changes
# ---------------------------------------------------------------------------


class TestDetectChanges:

    def test_새_파일_감지(self, make_config):
        tracker = IncrementalTracker(make_config())
        current = {"a.txt": "h1", "b.txt": "h2"}
        saved: dict[str, str] = {}

        new, modified, deleted = tracker.detect_changes(current, saved)

        assert new == ["a.txt", "b.txt"]
        assert modified == []
        assert deleted == []

    def test_수정_파일_감지(self, make_config):
        tracker = IncrementalTracker(make_config())
        current = {"a.txt": "new_hash"}
        saved = {"a.txt": "old_hash"}

        new, modified, deleted = tracker.detect_changes(current, saved)

        assert new == []
        assert modified == ["a.txt"]
        assert deleted == []

    def test_삭제_파일_감지(self, make_config):
        tracker = IncrementalTracker(make_config())
        current: dict[str, str] = {}
        saved = {"a.txt": "h1"}

        new, modified, deleted = tracker.detect_changes(current, saved)

        assert new == []
        assert modified == []
        assert deleted == ["a.txt"]

    def test_복합_변경_감지(self, make_config):
        tracker = IncrementalTracker(make_config())
        current = {"a.txt": "same", "b.txt": "changed", "d.txt": "new_file"}
        saved = {"a.txt": "same", "b.txt": "original", "c.txt": "removed"}

        new, modified, deleted = tracker.detect_changes(current, saved)

        assert new == ["d.txt"]
        assert modified == ["b.txt"]
        assert deleted == ["c.txt"]

    def test_변경_없음(self, make_config):
        tracker = IncrementalTracker(make_config())
        hashes = {"a.txt": "h1", "b.txt": "h2"}

        new, modified, deleted = tracker.detect_changes(hashes, hashes)

        assert new == []
        assert modified == []
        assert deleted == []


# ---------------------------------------------------------------------------
# merge_qa_pairs
# ---------------------------------------------------------------------------


class TestMergeQAPairs:

    def test_append_전략(self, make_config, make_qa_pair):
        tracker = IncrementalTracker(make_config())
        existing = [make_qa_pair(question="Q1")]
        new_pairs = [make_qa_pair(question="Q2")]

        merged = tracker.merge_qa_pairs(existing, new_pairs, "append")

        questions = [p.question for p in merged]
        assert "Q1" in questions
        assert "Q2" in questions

    def test_replace_전략(self, make_config, make_qa_pair):
        tracker = IncrementalTracker(make_config())
        existing = [make_qa_pair(question="Q1")]
        new_pairs = [make_qa_pair(question="Q2")]

        merged = tracker.merge_qa_pairs(existing, new_pairs, "replace")

        questions = [p.question for p in merged]
        assert "Q1" not in questions
        assert "Q2" in questions

    def test_중복_제거(self, make_config, make_qa_pair):
        tracker = IncrementalTracker(make_config())
        existing = [make_qa_pair(question="Q1")]
        new_pairs = [make_qa_pair(question="Q1"), make_qa_pair(question="Q2")]

        merged = tracker.merge_qa_pairs(existing, new_pairs, "append")

        questions = [p.question for p in merged]
        assert questions.count("Q1") == 1
        assert "Q2" in questions

    def test_빈_기존_쌍(self, make_config, make_qa_pair):
        tracker = IncrementalTracker(make_config())
        new_pairs = [make_qa_pair(question="Q1")]

        merged = tracker.merge_qa_pairs([], new_pairs, "append")

        assert len(merged) == 1

    def test_빈_새_쌍(self, make_config, make_qa_pair):
        tracker = IncrementalTracker(make_config())
        existing = [make_qa_pair(question="Q1")]

        merged = tracker.merge_qa_pairs(existing, [], "append")

        assert len(merged) == 1


# ---------------------------------------------------------------------------
# get_changed_files
# ---------------------------------------------------------------------------


class TestGetChangedFiles:

    def test_첫_실행_모든_파일_반환(self, make_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (docs_dir / "a.txt").write_text("content a", encoding="utf-8")
        (docs_dir / "b.txt").write_text("content b", encoding="utf-8")

        config = make_config(
            paths={"documents": str(docs_dir), "output": str(output_dir)},
            parsing={"formats": ["txt"]},
        )
        tracker = IncrementalTracker(config)
        changed = tracker.get_changed_files(docs_dir)

        names = [f.name for f in changed]
        assert "a.txt" in names
        assert "b.txt" in names

    def test_변경_없으면_빈_리스트(self, make_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (docs_dir / "a.txt").write_text("content a", encoding="utf-8")

        config = make_config(
            paths={"documents": str(docs_dir), "output": str(output_dir)},
            parsing={"formats": ["txt"]},
        )
        tracker = IncrementalTracker(config)
        tracker.get_changed_files(docs_dir)
        tracker.commit_hashes()
        changed = tracker.get_changed_files(docs_dir)

        assert changed == []

    def test_파일_수정_후_감지(self, make_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        doc_file = docs_dir / "a.txt"
        doc_file.write_text("version 1", encoding="utf-8")

        config = make_config(
            paths={"documents": str(docs_dir), "output": str(output_dir)},
            parsing={"formats": ["txt"]},
        )
        tracker = IncrementalTracker(config)
        tracker.get_changed_files(docs_dir)
        tracker.commit_hashes()

        doc_file.write_text("version 2", encoding="utf-8")
        changed = tracker.get_changed_files(docs_dir)

        assert len(changed) == 1
        assert changed[0].name == "a.txt"

    def test_해시_파일_저장_확인(self, make_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (docs_dir / "a.txt").write_text("data", encoding="utf-8")

        config = make_config(
            paths={"documents": str(docs_dir), "output": str(output_dir)},
            parsing={"formats": ["txt"]},
        )
        tracker = IncrementalTracker(config)
        tracker.get_changed_files(docs_dir)
        tracker.commit_hashes()

        hash_file = output_dir / config.incremental.hash_file
        assert hash_file.is_file()
        saved = json.loads(hash_file.read_text(encoding="utf-8"))
        assert "a.txt" in saved

    def test_반환값은_Path_객체(self, make_config, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (docs_dir / "a.txt").write_text("data", encoding="utf-8")

        config = make_config(
            paths={"documents": str(docs_dir), "output": str(output_dir)},
            parsing={"formats": ["txt"]},
        )
        tracker = IncrementalTracker(config)
        changed = tracker.get_changed_files(docs_dir)

        assert all(isinstance(f, Path) for f in changed)
