"""allganize/RAG-Evaluation-Dataset-KO의 원본 PDF best-effort 다운로드.

documents.csv를 HuggingFace에서 가져와 도메인 필터링 후:
  1) URL이 .pdf 직링크면 곧장 다운로드
  2) HTML viewer 페이지면 bs4로 첨부 PDF anchor 탐색
  3) 둘 다 실패하면 manual_download_list.txt에 (file_name, url)을 적고 사용자에게 안내

사용:
    uv run python scripts/fetch_allganize_pdfs.py \\
        --domains public finance law \\
        --out-dir allganize-eval-project/documents

대상 디렉토리에 평면 구조로 PDF를 저장 (expected_doc 매칭을 위해 원본 file_name 유지).
PDF 재배포는 금지되므로 이 스크립트는 사용자가 로컬에서 실행하는 패턴 — 결과 PDF를 repo에 commit하지 말 것.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

DOCUMENTS_CSV_URL = (
    "https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO/"
    "resolve/main/documents.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
}


def load_documents_csv() -> list[dict]:
    resp = httpx.get(DOCUMENTS_CSV_URL, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


def looks_like_pdf(content_type: str, body: bytes) -> bool:
    if "pdf" in content_type.lower():
        return True
    return body[:4] == b"%PDF"


def try_direct_pdf(client: httpx.Client, url: str) -> bytes | None:
    """URL이 PDF 직링크인지 시도 — Content-Type이나 magic byte로 판정."""
    try:
        r = client.get(url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        if looks_like_pdf(r.headers.get("content-type", ""), r.content):
            return r.content
    except Exception:
        return None
    return None


SCOURT_DOWNLOAD_RE = re.compile(
    r"""javascript:download\(\s*['"]([^'"]+)['"]\s*,\s*['"]([^'"]+)['"]\s*\)""",
    re.IGNORECASE,
)
KIF_DOWNLOAD_RE = re.compile(
    r"""location\.href\s*=\s*['"](https://www\.kif\.re\.kr/[^'"]+)['"]""",
    re.IGNORECASE,
)
MPM_DOWNLOAD_RE = re.compile(
    r"""Jnit_boardDownload\(\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def try_kif_flexer(client: httpx.Client, page_url: str) -> bytes | None:
    """vwserver.kif.re.kr/flexer/viewer.jsp 페이지에서 Download 버튼의
    `location.href='https://www.kif.re.kr/kif4/publication/viewer?...'` URL을 추출해 다운로드."""
    if "kif.re.kr" not in urlparse(page_url).netloc:
        return None
    try:
        r = client.get(page_url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        m = KIF_DOWNLOAD_RE.search(r.text)
        if not m:
            return None
        post = client.get(
            m.group(1),
            timeout=120.0,
            follow_redirects=True,
            headers={"Referer": page_url},
        )
        post.raise_for_status()
        if looks_like_pdf(post.headers.get("content-type", ""), post.content):
            return post.content
    except Exception:
        return None
    return None


def try_mois_node40(client: httpx.Client, page_url: str) -> bytes | None:
    """mois.go.kr 게시판은 멀티-노드 백엔드에서 일부 노드만 본문을 렌더한다 (node40만 동작 확인됨).
    경로에 `;jsessionid=...node40` 더미 세그먼트를 추가해 라우팅을 강제한 뒤,
    /cmm/fms/FileDown.do?atchFileId=...&fileSn=... 첨부 다운로드 링크를 추출."""
    if "mois.go.kr" not in urlparse(page_url).netloc:
        return None
    try:
        parsed = urlparse(page_url)
        # `path;jsessionid=DummyForRoute.node40` — JEE 컨테이너 노드 핀
        sticky_path = parsed.path + ";jsessionid=DummyForRoute.node40"
        sticky_url = parsed._replace(path=sticky_path).geturl()
        r = client.get(sticky_url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        m = re.search(
            r"""href=['"](/cmm/fms/FileDown\.do[^'"]+)['"]""", r.text
        )
        if not m:
            return None
        dl_url = urljoin("https://www.mois.go.kr", m.group(1).replace("&amp;", "&"))
        post = client.get(
            dl_url,
            timeout=120.0,
            follow_redirects=True,
            headers={"Referer": sticky_url},
        )
        post.raise_for_status()
        if looks_like_pdf(post.headers.get("content-type", ""), post.content):
            return post.content
    except Exception:
        return None
    return None


def try_mpm_board(client: httpx.Client, page_url: str) -> bytes | None:
    """mpm.go.kr 게시판 페이지에서 `Jnit_boardDownload('/board/file/...')` 첫 인자를 추출해 다운로드."""
    if "mpm.go.kr" not in urlparse(page_url).netloc:
        return None
    try:
        r = client.get(page_url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        m = MPM_DOWNLOAD_RE.search(r.text)
        if not m:
            return None
        post = client.get(
            urljoin("https://www.mpm.go.kr", m.group(1)),
            timeout=120.0,
            follow_redirects=True,
            headers={"Referer": page_url},
        )
        post.raise_for_status()
        if looks_like_pdf(post.headers.get("content-type", ""), post.content):
            return post.content
    except Exception:
        return None
    return None


def try_scourt_attach(client: httpx.Client, page_url: str) -> bytes | None:
    """scourt.go.kr DcNewsView 페이지에서 JS `download(file, name)` 호출을 추출해
    file.scourt.go.kr/AttachDownload로 POST 다운로드를 시도."""
    if "scourt.go.kr" not in urlparse(page_url).netloc:
        return None
    try:
        r = client.get(page_url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        html = r.content.decode("euc-kr", errors="replace")
        m = SCOURT_DOWNLOAD_RE.search(html)
        if not m:
            return None
        internal_file, display_name = m.group(1), m.group(2)
        post = client.post(
            "https://file.scourt.go.kr/AttachDownload",
            data={"file": internal_file, "path": "003", "downFile": display_name},
            timeout=60.0,
            follow_redirects=True,
        )
        post.raise_for_status()
        if looks_like_pdf(post.headers.get("content-type", ""), post.content):
            return post.content
    except Exception:
        return None
    return None


def try_html_scrape(client: httpx.Client, page_url: str) -> bytes | None:
    """HTML 페이지에서 PDF anchor·iframe·script 패턴을 탐색해 다운로드 시도."""
    try:
        r = client.get(page_url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        if not r.text:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        candidates: list[str] = []
        for tag in soup.find_all(["a", "iframe"]):
            href = tag.get("href") or tag.get("src")
            if not href:
                continue
            if ".pdf" in href.lower() or "download" in href.lower() or "file" in href.lower():
                candidates.append(urljoin(page_url, href))
        for cand in candidates:
            body = try_direct_pdf(client, cand)
            if body:
                return body
    except Exception:
        return None
    return None


def safe_filename(name: str) -> str:
    # 파일명 자체는 유지 (expected_doc 매칭). 단 경로 분리자만 sanitize.
    return name.replace("/", "_").replace("\\", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["public", "finance", "law"],
    )
    parser.add_argument(
        "--out-dir",
        default="allganize-eval-project/documents",
        help="PDF 저장 디렉토리 (평면 구조 — expected_doc 매칭을 위해 원본 file_name 유지)",
    )
    parser.add_argument(
        "--manual-list",
        default="allganize-eval-project/manual_download_list.txt",
        help="자동 다운로드 실패 시 사용자가 수동으로 다운로드해야 할 (file_name, url) 목록을 적을 경로",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="요청간 sleep(초)")
    args = parser.parse_args()

    docs = load_documents_csv()
    target = [d for d in docs if d["domain"] in args.domains]
    print(f"[total] {len(target)} PDFs in domains={args.domains}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manual_path = Path(args.manual_list)
    manual_path.parent.mkdir(parents=True, exist_ok=True)

    successes: list[dict] = []
    failures: list[dict] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        for i, row in enumerate(target, 1):
            file_name = row["file_name"]
            url = row["url"]
            domain = row["domain"]
            dest = out_dir / safe_filename(file_name)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"[{i:3}/{len(target)}] [skip] already exists: {dest.name}", file=sys.stderr)
                successes.append(row)
                continue

            print(f"[{i:3}/{len(target)}] [{domain}] {file_name}", file=sys.stderr)
            body = try_direct_pdf(client, url)
            if not body:
                body = try_scourt_attach(client, url)
            if not body:
                body = try_kif_flexer(client, url)
            if not body:
                body = try_mpm_board(client, url)
            if not body:
                body = try_mois_node40(client, url)
            if not body:
                body = try_html_scrape(client, url)

            if body:
                dest.write_bytes(body)
                print(f"           [OK] {len(body)//1024} KB saved", file=sys.stderr)
                successes.append(row)
            else:
                print(f"           [FAIL] manual download needed", file=sys.stderr)
                failures.append(row)
            time.sleep(args.delay)

    if failures:
        lines = [
            "# Manual download required — automatic fetch failed for these.",
            "# Save each PDF to the out-dir keeping the exact file_name (so expected_doc matches).",
            f"# out-dir: {out_dir}",
            "",
        ]
        for row in failures:
            lines.append(f"[{row['domain']}] {row['file_name']}")
            lines.append(f"    {row['url']}")
            lines.append("")
        manual_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"\n[summary] success={len(successes)} fail={len(failures)} out={out_dir}",
        file=sys.stderr,
    )
    if failures:
        print(f"[manual] see {manual_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
