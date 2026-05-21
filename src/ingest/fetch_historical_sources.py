"""
Historical Source Fetcher
=========================

Pull attested validator texts from explicit source manifests into
`data/raw/historical/<corpus_id>/`.

This is intentionally conservative:

- one row per source text
- explicit corpus routing
- no implicit scraping across sites
- designed primarily for public-domain Wikisource pages
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_HISTORICAL_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
DEFAULT_SOURCE_MANIFEST = RAW_HISTORICAL_DIR / "validator_sources.csv"


@dataclass(frozen=True)
class SourceRow:
    corpus_id: str
    title: str
    source_url: str
    fetch_mode: str
    output_filename: str
    status: str
    notes: str
    start_marker: str
    end_marker: str


class _WikisourceHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_stack: list[str] = []
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        klass = attrs_dict.get("class", "")
        if tag in {"style", "script", "table", "sup", "figure", "math"}:
            self._skip_stack.append(tag)
            return
        if "reference" in klass or "mw-editsection" in klass or "navigation-not-searchable" in klass:
            self._skip_stack.append(tag)
            return
        if tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "br", "hr"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            top = self._skip_stack[-1]
            if top == tag:
                self._skip_stack.pop()
                return
        if tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)
            self._chunks.append(" ")

    def text(self) -> str:
        raw = unescape("".join(self._chunks))
        raw = re.sub(r"\[[0-9]+\]", "", raw)
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]{2,}", " ", raw)
        cleaned_lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue
            if stripped in {"*", "* * *", "Télécharger", "Outils"}:
                continue
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines).strip() + "\n"


def _load_rows(manifest_path: Path) -> list[SourceRow]:
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            SourceRow(
                corpus_id=row["corpus_id"].strip(),
                title=row["title"].strip(),
                source_url=row["source_url"].strip(),
                fetch_mode=row["fetch_mode"].strip(),
                output_filename=row["output_filename"].strip(),
                status=row["status"].strip(),
                notes=row.get("notes", "").strip(),
                start_marker=row.get("start_marker", "").strip(),
                end_marker=row.get("end_marker", "").strip(),
            )
            for row in reader
        ]


def _extract_wikisource_page_title(source_url: str) -> tuple[str, str]:
    parsed = urlparse(source_url)
    host = parsed.netloc
    if not host.endswith("wikisource.org"):
        raise ValueError(f"Unsupported Wikisource host: {host}")
    if parsed.path.startswith("/wiki/"):
        title = unquote(parsed.path.split("/wiki/", 1)[1])
        return host, title
    query = parse_qs(parsed.query)
    if "title" in query and query["title"]:
        return host, query["title"][0]
    raise ValueError(f"Could not determine Wikisource page title from URL: {source_url}")


def _fetch_url_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "ProjectRBT-HistoricalFetcher/1.0 (+local research ingestion)",
        },
    )
    with urlopen(request) as response:
        payload = response.read()
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            charset = response.headers.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")


def fetch_wikisource_page(source_url: str) -> str:
    host, title = _extract_wikisource_page_title(source_url)
    api_url = (
        f"https://{host}/w/api.php?"
        + urlencode(
            {
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "formatversion": "2",
            }
        )
    )
    payload = json.loads(_fetch_url_text(api_url))
    html = payload["parse"]["text"]
    parser = _WikisourceHTMLTextExtractor()
    parser.feed(html)
    return parser.text()


def fetch_source(row: SourceRow, output_root: Path = RAW_HISTORICAL_DIR) -> Path:
    output_dir = output_root / row.corpus_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / row.output_filename
    if row.fetch_mode == "wikisource_page":
        text = fetch_wikisource_page(row.source_url)
    else:
        raise ValueError(f"Unsupported fetch_mode: {row.fetch_mode}")
    if row.start_marker:
        idx = text.find(row.start_marker)
        if idx >= 0:
            text = text[idx:]
    if row.end_marker:
        idx = text.find(row.end_marker)
        if idx >= 0:
            text = text[:idx]
    output_path.write_text(text, encoding="utf-8")
    log.info("Wrote %s", output_path)
    return output_path


def fetch_all(
    manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    corpus_ids: set[str] | None = None,
    include_statuses: set[str] | None = None,
) -> list[Path]:
    rows = _load_rows(manifest_path)
    selected = []
    for row in rows:
        if corpus_ids and row.corpus_id not in corpus_ids:
            continue
        if include_statuses and row.status not in include_statuses:
            continue
        selected.append(row)
    outputs = []
    for row in selected:
        outputs.append(fetch_source(row))
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch attested validator texts into data/raw/historical")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--corpus", action="append", default=[], help="Limit to one or more corpus ids")
    parser.add_argument(
        "--status",
        action="append",
        default=["ready"],
        help="Limit to one or more source statuses (default: ready)",
    )
    args = parser.parse_args()
    corpus_ids = set(args.corpus) if args.corpus else None
    statuses = set(args.status) if args.status else None
    outputs = fetch_all(args.manifest, corpus_ids=corpus_ids, include_statuses=statuses)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
