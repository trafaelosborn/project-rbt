from src.ingest.fetch_historical_sources import _WikisourceHTMLTextExtractor
from pathlib import Path


def test_wikisource_html_extractor_strips_reference_and_tables():
    html = """
    <div class="mw-parser-output">
      <h2>Title</h2>
      <p>First line<sup class="reference">[1]</sup>.</p>
      <table><tr><td>drop me</td></tr></table>
      <p>Second line.</p>
    </div>
    """
    parser = _WikisourceHTMLTextExtractor()
    parser.feed(html)
    text = parser.text()
    assert "Title" in text
    assert "First line" in text
    assert "Second line." in text
    assert "[1]" not in text
    assert "drop me" not in text


def test_source_manifest_supports_optional_markers():
    csv_path = Path(__file__).resolve().parent / "_tmp_fetch_sources.csv"
    try:
        csv_path.write_text(
            "corpus_id,title,source_url,fetch_mode,output_filename,status,notes,start_marker,end_marker\n"
            "old_spanish,Poema,x,wikisource_page,a.txt,ready,note,POEMA,FIN\n",
            encoding="utf-8",
        )
        from src.ingest.fetch_historical_sources import _load_rows

        rows = _load_rows(csv_path)
        assert rows[0].start_marker == "POEMA"
        assert rows[0].end_marker == "FIN"
    finally:
        csv_path.unlink(missing_ok=True)
