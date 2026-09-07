"""Conversion tests: corpus rows -> typed app_knowledge items. Hermetic, no network."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from erb_hydradb import corpus
from erb_hydradb.convert import SOURCE_TYPE_TO_KIND_PROVIDER, TITLE_MAX, convert

ERB_REPO = Path(__file__).parent / "data" / "erb_repo"
DB, COLL = "erb_test_db", "entire"
# What the scorer strips to recover the benchmark document id.
SUFFIX_RE = re.compile(r"_(m\d{4}|e\d{2}|c\d{2})$")
SCALAR = (str, int, float, bool, type(None))


@pytest.fixture(scope="module")
def docs() -> dict[str, dict]:
    """source_type -> corpus row, built through the same path the real corpus uses."""
    rows = {}
    for doc_id, st, title, content, content_len in corpus.iter_repo_rows(ERB_REPO):
        rows[st] = {"doc_id": doc_id, "source_type": st, "title": title,
                    "content": content, "content_len": content_len}
    assert set(rows) == {"slack", "gmail", "jira", "confluence", "hubspot"}
    return rows


@pytest.fixture(scope="module")
def items_by_source(docs) -> dict[str, list[dict]]:
    return {st: convert(doc, DB, COLL) for st, doc in docs.items()}


# -- mapping ------------------------------------------------------------------
@pytest.mark.parametrize("source_type,kinds,provider", [
    ("slack", {"message"}, "slack"),
    ("gmail", {"email"}, "gmail"),
    ("jira", {"ticket", "comment"}, "jira"),
    ("confluence", {"knowledge_base"}, "notion"),
    ("hubspot", {"custom"}, "hubspot"),
])
def test_kinds_and_providers_per_source(items_by_source, source_type, kinds, provider):
    items = items_by_source[source_type]
    assert items, source_type
    assert {it["kind"] for it in items} == kinds
    assert {it["provider"] for it in items} == {provider}
    assert {it["type"] for it in items} == {source_type}


def test_mapping_table_covers_every_benchmark_source():
    assert set(SOURCE_TYPE_TO_KIND_PROVIDER) == {
        "slack", "gmail", "jira", "linear", "github", "confluence", "google_drive", "fireflies", "hubspot"}
    assert SOURCE_TYPE_TO_KIND_PROVIDER["confluence"] == ("knowledge_base", "notion")


# -- item invariants ----------------------------------------------------------
def test_every_item_is_well_formed(items_by_source, docs):
    for st, items in items_by_source.items():
        for it in items:
            for key in ("id", "kind", "provider", "title", "fields", "external_id", "metadata"):
                assert key in it, (st, key)
            assert it["fields"]["kind"] == it["kind"]
            assert it["tenant_id"] == DB and it["sub_tenant_id"] == COLL
            assert it["external_id"] == it["id"]
            assert 0 < len(it["title"]) <= TITLE_MAX
            assert None not in it.values()
            assert it["metadata"]["erb_doc_id"] == docs[st]["doc_id"]
            assert it["metadata"]["erb_source_type"] == st
            # metadata is flat: scalars, or lists of scalars
            for v in it["metadata"].values():
                assert isinstance(v, SCALAR) or (isinstance(v, list) and all(isinstance(x, SCALAR) for x in v))
            assert SUFFIX_RE.sub("", it["id"]) == docs[st]["doc_id"]


def test_ids_are_stable_across_calls(docs):
    for doc in docs.values():
        a = json.dumps(convert(doc, DB, COLL), sort_keys=True)
        b = json.dumps(convert(doc, DB, COLL), sort_keys=True)
        assert a == b


def test_empty_document_yields_no_items():
    assert convert({"doc_id": "dsid_0", "source_type": "slack", "title": "x", "content": "  \n"}, DB, COLL) == []


def test_title_is_capped():
    long_title = "T" * 3000
    [it] = convert({"doc_id": "dsid_1", "source_type": "confluence", "title": long_title, "content": "body"},
                   DB, COLL)
    assert len(it["title"]) == TITLE_MAX and it["title"].endswith("…")
    assert it["fields"]["title"] == it["title"]


# -- slack --------------------------------------------------------------------
def test_slack_thread_becomes_chained_messages(items_by_source, docs):
    items = items_by_source["slack"]
    doc_id = docs["slack"]["doc_id"]
    n = len(items)
    assert n == 16  # the sample thread has 16 "Name: text" lines
    root = f"{doc_id}_m0000"
    for i, it in enumerate(items):
        f = it["fields"]
        assert it["id"] == f"{doc_id}_m{i:04d}"
        assert f["thread_id"] == root
        assert f["parent_id"] == ("" if i == 0 else f"{doc_id}_m{i - 1:04d}")
        assert f["author"] and f["body"]
        assert it["title"] == f"#eng-infra — message {i + 1}/{n}"
        assert it["metadata"]["channel"] == "eng-infra"
    # synthetic, deterministic timestamps: 2025-01-01T00:00:00Z + 45 s per message
    assert items[0]["timestamp"] == "2025-01-01T00:00:00Z"
    assert items[1]["timestamp"] == "2025-01-01T00:00:45Z"
    assert items[0]["fields"]["author"] == "Alex"
    assert items[0]["fields"]["body"].startswith("Quick cost comp across regions")
    # a fenced code block stays inside the message that introduced it
    code_msg = [it for it in items if "```" in it["fields"]["body"]]
    assert len(code_msg) == 1 and code_msg[0]["fields"]["author"] == "Alex"


# -- gmail --------------------------------------------------------------------
def test_gmail_thread_is_split_into_reply_chain(items_by_source, docs):
    items = items_by_source["gmail"]
    doc_id = docs["gmail"]["doc_id"]
    assert len(items) == 6
    for i, it in enumerate(items):
        f = it["fields"]
        assert it["id"] == f"{doc_id}_e{i:02d}"
        assert f["thread_id"] == doc_id
        assert f["from"] and f["subject"] and f["body"]
        assert "reply_to_id" not in f  # renamed on the wire
        if i == 0:
            assert "in_reply_to" not in f
        else:
            assert f["in_reply_to"] == f"{doc_id}_e{i - 1:02d}"
        assert it["metadata"]["thread_index"] == i and it["metadata"]["thread_size"] == 6
    assert items[0]["timestamp"] == "2025-06-10T09:12:00"
    assert items[0]["fields"]["created_at"] == items[0]["timestamp"]
    assert items[0]["fields"]["to"] and isinstance(items[0]["fields"]["to"], list)


def test_single_message_gmail_keeps_bare_doc_id():
    content = "From: A <a@x.test>\nTo: B <b@x.test>\nSubject: Hi\n\nBody text"
    [it] = convert({"doc_id": "dsid_2", "source_type": "gmail", "title": "Hi", "content": content}, DB, COLL)
    assert it["id"] == "dsid_2" and it["fields"]["to"] == ["B <b@x.test>"]


# -- tickets ------------------------------------------------------------------
def test_ticket_with_comments(items_by_source, docs):
    items = items_by_source["jira"]
    doc_id = docs["jira"]["doc_id"]
    ticket, comments = items[0], items[1:]
    assert ticket["id"] == doc_id and ticket["kind"] == "ticket"
    assert ticket["metadata"]["comment_count"] == len(comments) == 9
    assert ticket["fields"]["description"].startswith("description:\n")
    assert "\ncomments:\n" not in ticket["fields"]["description"]
    # trailing sections after the comments block are kept with the ticket
    assert "\nresolution:\n" in ticket["fields"]["description"]
    for i, c in enumerate(comments):
        f = c["fields"]
        assert c["id"] == f"{doc_id}_c{i:02d}" and c["kind"] == "comment"
        assert f["parent_id"] == doc_id
        assert f["author"] and f["body"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T00:00:00Z", f["created_at"])
        assert c["metadata"]["comment_index"] == i
        assert c["title"].startswith("Comment on: ")
    assert comments[0]["fields"]["author"] == "Liam O'Rourke, Support"
    assert comments[0]["fields"]["created_at"] == "2025-05-07T00:00:00Z"


# -- confluence / hubspot -----------------------------------------------------
def test_confluence_revisions_go_to_metadata(items_by_source):
    [it] = items_by_source["confluence"]
    assert it["kind"] == "knowledge_base" and it["provider"] == "notion"
    md = it["metadata"]
    assert md["revision_count"] == 2 and len(md["revisions"]) == 2
    for rev in md["revisions"]:  # flattened to "date: note" strings for the API's depth-1 metadata
        assert isinstance(rev, str) and re.match(r"^\d{4}-\d{2}-\d{2}: ", rev)
    assert it["timestamp"] == "2025-02-03T00:00:00Z"
    assert it["fields"]["body"] and it["fields"]["title"] == it["title"]


def test_hubspot_is_custom_with_data(items_by_source, docs):
    [it] = items_by_source["hubspot"]
    assert it["kind"] == "custom" and it["provider"] == "hubspot"
    data = it["fields"]["data"]
    assert data["title"] == docs["hubspot"]["title"]
    assert data["body"] == docs["hubspot"]["content"]
    assert "timestamp" not in it
