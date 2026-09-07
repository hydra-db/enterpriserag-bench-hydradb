"""Convert one benchmark document into typed HydraDB ``app_knowledge`` items.

HydraDB ingests application content via ``POST /context/ingest``
(``type=knowledge``) with an ``app_knowledge`` array. Every item uses the
*typed* schema, so HydraDB routes it through its typed handlers (actor
extraction, thread/reply graph edges, thread-ancestor inference) instead of
treating it as a generic document::

    {
      "id": str,                    # stable id == upsert key
      "tenant_id": str,             # the database
      "sub_tenant_id": str,         # the collection
      "title": str,                 # <= 1000 chars
      "kind": str,                  # email | message | ticket | knowledge_base | comment | custom
      "provider": str,              # gmail | slack | jira | linear | github | notion | ...
      "external_id": str,
      "fields": {"kind": <same as kind>, ...typed body fields...},
      "metadata": {...},            # flat: scalars, or lists of scalars/strings
      "type": str,                  # the benchmark source_type (display/analytics)
      "timestamp": str,             # optional, ISO-8601
    }

Mapping (benchmark ``source_type`` -> HydraDB ``kind`` / ``provider``):

    source_type    kind            provider       items per document
    -----------    -------------   ------------   -------------------------------------------
    slack          message         slack          one per parsed message, ids <doc_id>_m0000..
    gmail          email           gmail          one per parsed email; ids <doc_id>_e00.. when
                                                  the thread has >1 message, else <doc_id>
    jira           ticket+comment  jira           <doc_id> (ticket) + <doc_id>_c00.. (comments)
    linear         ticket+comment  linear         same
    github         ticket+comment  github         same
    confluence     knowledge_base  notion         one, id <doc_id> (Confluence is ingested as a
                                                  Notion substitute)
    google_drive   knowledge_base  google_drive   one, id <doc_id>
    fireflies      knowledge_base  fireflies      one, id <doc_id>
    hubspot        custom          hubspot        one, id <doc_id>
    (empty content)                               no items

Every item's ``id`` is the benchmark ``dsid_...`` document id, optionally with a
typed suffix (``_mNNNN`` message, ``_eNN`` email, ``_cNN`` comment). The
scoring side strips the suffix to recover the document id, so a hit on any
part of a document counts for that document.

Conversion choices a reader must know:

* **Slack timestamps are synthetic.** The benchmark's Slack transcripts carry
  no per-message time, so message ``i`` gets ``2025-01-01T00:00:00Z + 45 s * i``
  (deterministic; only ordering matters). ``fields.thread_id`` is the first
  message's id on every message and ``fields.parent_id`` is the previous
  message's id, which drives HydraDB's thread-member / reply-to edges.
* **Gmail threads are split into one ``email`` item per message.** Headers
  (From/To/Cc/Date/Subject) are parsed from the RFC-style text; the reply
  pointer to the previous message is sent as ``fields.in_reply_to``.
* **Tickets keep their description; comments become ``comment`` items** with
  ``fields.parent_id`` = the ticket id. Dated comments get ``created_at``.
* **Confluence is ingested with provider ``notion``.** For Confluence and
  Google Drive pages, a revision-history section is parsed into
  ``metadata.revisions`` (``"<date>: <note>"`` strings) and the item's
  ``timestamp`` is the latest revision date. No prior page versions are
  synthesized.
* **HubSpot records are ``custom`` items** whose ``fields.data`` carries the
  title plus either the whole body or one key per labelled section.
* **Titles are capped at 1000 characters** (the server's title column holds
  1024; an over-long title fails the whole batch).
* Literal ``\\n`` / ``\\t`` sequences that dominate a document (escaped-export
  artifacts in gmail / drive / confluence / fireflies) are turned into real
  whitespace before parsing.

This module is a faithful port of the converter that produced the published
result; the parsing heuristics are intentionally unchanged.
"""

from __future__ import annotations

import ast
import json
import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime

# --- source_type -> (kind, provider) ----------------------------------------
SOURCE_TYPE_TO_KIND_PROVIDER: dict[str, tuple[str, str]] = {
    "slack": ("message", "slack"),
    "gmail": ("email", "gmail"),
    "jira": ("ticket", "jira"),
    "linear": ("ticket", "linear"),
    "github": ("ticket", "github"),
    "confluence": ("knowledge_base", "notion"),
    "google_drive": ("knowledge_base", "google_drive"),
    "fireflies": ("knowledge_base", "fireflies"),
    "hubspot": ("custom", "hubspot"),
}

TITLE_MAX = 1000
CHANNEL_MAX = 80

# Deterministic base epoch for synthetic Slack timestamps (2025-01-01T00:00:00Z).
_SLACK_BASE_EPOCH = 1_735_689_600
_SLACK_STEP_SECONDS = 45

# A line that introduces a new speaker: "<Name>: <text>" or "<Name> (Role): <text>".
_AUTHOR_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ._'\-]{0,29}?(?:\s\([A-Za-z][A-Za-z0-9 /&.\-]{0,24}\))?):\s?(.*)$")


def _cap(s: str | None, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _iso(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# Log/section keywords that appear as "word: value" lines inside chat messages
# and must never be read as speakers.
_NON_AUTHOR_KEYWORDS = frozenset({
    "observed", "expected", "result", "results", "repro", "note", "notes",
    "error", "errors", "warning", "info", "context", "status", "update",
    "summary", "action", "actions", "impact", "cause", "fix", "workaround",
    "output", "input", "steps", "goal", "plan", "tldr", "details", "example",
    "examples", "question", "answer", "todo", "owner", "eta", "severity",
    "priority", "rotation", "policy", "metric", "metrics", "latency", "url",
    "link", "ref", "source", "target", "config", "env", "version", "date",
    "time", "subject", "from", "to", "cc",
})


def _looks_like_author(name: str) -> bool:
    toks = name.split()
    if not (1 <= len(toks) <= 4 and any(c.isalpha() for c in name)):
        return False
    return name.strip().lower() not in _NON_AUTHOR_KEYWORDS


def _typed_item(*, item_id: str, database: str, collection: str, title: str, kind: str,
                provider: str, external_id: str, fields: dict, metadata: dict | None = None,
                timestamp: str | None = None) -> dict:
    """Build one typed app_knowledge item. ``fields.kind`` must equal ``kind``."""
    item = {
        "id": item_id,
        "tenant_id": database,
        "sub_tenant_id": collection,
        "title": _cap(title, TITLE_MAX),
        "kind": kind,
        "provider": provider,
        "external_id": external_id,
        "fields": fields,
        "metadata": metadata or {},
    }
    src_type = (metadata or {}).get("erb_source_type")
    if src_type:
        item["type"] = src_type
    if timestamp:
        item["timestamp"] = timestamp
    return item


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------
def parse_slack_messages(content: str) -> list[dict]:
    """Split a Slack transcript into ordered messages.

    Robust to multi-line messages, fenced code blocks (whose ``key: value``
    lines must not be read as new speakers) and bot lines. A new message starts
    only on a non-fenced line that matches the author pattern and is preceded
    by a blank line (or is the first line) -- unless the export is "dense"
    (nearly every line is ``Name: text``), in which case every author line
    starts a message. A speaker already seen in the thread may also start a new
    message without a blank separator.
    """
    _fence = False
    n_candidates = n_nonblank = 0
    for line in content.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            _fence = not _fence
            continue
        if not s or _fence:
            continue
        n_nonblank += 1
        m = _AUTHOR_RE.match(line)
        if m and _looks_like_author(m.group(1)):
            n_candidates += 1
    dense = n_candidates >= 3 and n_candidates / max(n_nonblank, 1) >= 0.6

    messages: list[dict] = []
    cur: dict | None = None
    in_fence = False
    prev_blank = True
    seen_authors: set[str] = set()

    for line in content.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            if cur is not None:
                cur["_lines"].append(line)
            prev_blank = False
            continue

        candidate = _AUTHOR_RE.match(line) if not in_fence else None
        allowed = prev_blank or dense or (
            candidate is not None
            and candidate.group(1).strip().lower() in seen_authors)
        m = candidate if allowed else None
        if m and _looks_like_author(m.group(1)):
            cur = {"author": m.group(1).strip(), "_lines": [m.group(2)]}
            seen_authors.add(cur["author"].lower())
            messages.append(cur)
        elif cur is not None:
            cur["_lines"].append(line)

        prev_blank = stripped == ""

    out = []
    for i, msg in enumerate(messages):
        text = "\n".join(msg["_lines"]).strip()
        code_blocks = re.findall(r"```.*?```", text, flags=re.DOTALL)
        out.append({
            "index": i, "author": msg["author"], "text": text,
            "has_code": bool(code_blocks), "code_blocks": code_blocks,
        })
    return out


def slack_to_items(doc: dict, database: str, collection: str) -> list[dict]:
    """One typed ``message`` item per parsed Slack message (thread root first)."""
    doc_id = doc["doc_id"]
    channel = _cap(doc.get("title") or "unknown", CHANNEL_MAX)
    msgs = parse_slack_messages(doc.get("content") or "")
    root_ext = f"{doc_id}_m0000"
    items: list[dict] = []

    for msg in msgs:
        i = msg["index"]
        ext = f"{doc_id}_m{i:04d}"
        epoch = _SLACK_BASE_EPOCH + i * _SLACK_STEP_SECONDS
        slack_ts = f"{epoch}.{i:06d}"
        fields = {
            "kind": "message",
            "body": msg["text"],
            "author": msg["author"],
            "thread_id": root_ext,
            "parent_id": "" if i == 0 else f"{doc_id}_m{i - 1:04d}",
            "url": f"https://slack.example/archives/{channel}/p{slack_ts.replace('.', '')}",
        }
        items.append(_typed_item(
            item_id=ext, database=database, collection=collection,
            title=f"#{channel} — message {i + 1}/{len(msgs)}",
            kind="message", provider="slack", external_id=ext, fields=fields,
            metadata={"channel": channel, "slack_ts": slack_ts, "erb_doc_id": doc_id,
                      "erb_source_type": "slack"},
            timestamp=_iso(epoch),
        ))
    return items


# ---------------------------------------------------------------------------
# Confluence / Google Drive — revision history -> structured metadata
# ---------------------------------------------------------------------------
_REV_HEADING_RE = re.compile(
    r"(revision\s+(history|log|notes)|change\s?log|change\s+history|"
    r"version\s+(history|notes)|draft\s+history|document\s+history|\bhistory\b)",
    re.I,
)
_REV_HEADING_EXCLUDE = re.compile(
    r"audit|\bci\b|versioning|compatib|observab|reporting|requirement", re.I
)
_REV_LINE_RE = re.compile(
    r"^[\-\*•#>\s]*(\d{4}[-/]\d{2}[-/]\d{2})\s*(?:\(([^)]{1,60})\))?\s*[:\-–—]\s*(.+?)\s*$"
)


def _is_revision_heading(line: str) -> bool:
    s = line.strip().lstrip("#>-*• ").rstrip(":").strip()
    return (
        3 <= len(s) <= 45
        and bool(_REV_HEADING_RE.search(s))
        and not _REV_HEADING_EXCLUDE.search(s)
    )


def parse_revisions(content: str) -> tuple[list[dict], str | None]:
    """Extract a page's edit history as ``[{date, note}]`` plus the latest date (ISO).

    Only dated lines under a revision-style heading count; the last heading
    that yields any dated line wins.
    """
    lines = content.split("\n")
    heading_idxs = [i for i, ln in enumerate(lines) if _is_revision_heading(ln)]
    if not heading_idxs:
        return [], None

    best: list[dict] = []
    heading_set = set(heading_idxs)
    for h in heading_idxs:
        revisions: list[dict] = []
        for j in range(h + 1, len(lines)):
            if j in heading_set:
                break
            m = _REV_LINE_RE.match(lines[j])
            if m:
                note = m.group(3).strip()
                if m.group(2):  # "(Author)" between date and note
                    note = f"{m.group(2)}: {note}"
                revisions.append({"date": m.group(1).replace("/", "-"), "note": note})
        if revisions:
            best = revisions

    latest = max((r["date"] for r in best), default=None)
    return best, (f"{latest}T00:00:00Z" if latest else None)


# ---------------------------------------------------------------------------
# Escaped-newline normalization
# ---------------------------------------------------------------------------
def normalize_escapes(text: str) -> str:
    """Turn literal ``\\n``/``\\t`` sequences into real whitespace when they
    dominate the text (an escaped-export artifact), leaving genuine code that
    merely mentions ``\\n`` untouched."""
    if not text:
        return text
    if text.count("\\\\n") >= 3:
        text = text.replace("\\\\r\\\\n", "\n").replace("\\\\n", "\n").replace("\\\\t", "    ")
    literal = text.count("\\n")
    real = text.count("\n")
    if literal >= 3 and literal > real:
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
    return text


# ---------------------------------------------------------------------------
# Gmail — a Python list literal of RFC-style emails, or plain RFC-style text
# ---------------------------------------------------------------------------
_EMAIL_HEADER_RE = re.compile(r"^([A-Za-z-]+):[ \t](.*)$", re.M)
_GMAIL_DATE_FORMATS = (
    "%a, %d %b %Y %H:%M:%S %z",          # RFC 2822: Tue, 21 Sep 2027 22:05:12 +0100
    "%a, %b %d, %Y at %I:%M %p",         # Gmail UI: Tue, Jun 3, 2025 at 9:12 AM
)


def _parse_gmail_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).isoformat()
    except Exception:  # noqa: BLE001
        pass
    for fmt in _GMAIL_DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return None


# An email boundary: a "From: ..." line after a blank line or a "---" rule.
_EMAIL_SPLIT_RE = re.compile(r"(?:\n\s*\n|\n\\?-{2,}\s*\n)(?=From:[ \t])")


def parse_gmail_messages(content: str) -> list[dict]:
    """Split raw gmail content into ordered ``{headers, body}`` messages.

    Handles a Python list literal of email strings (each element may hold
    several concatenated emails), plain text with one or more concatenated
    RFC-style emails, and escaped-newline exports.
    """
    raw_blocks: list[str]
    stripped = content.lstrip()
    if stripped.startswith("["):
        try:
            parsed = ast.literal_eval(content)
            raw_blocks = [str(m) for m in parsed] if isinstance(parsed, list) else [content]
        except (ValueError, SyntaxError):
            raw_blocks = [normalize_escapes(content).strip().lstrip("[\"'").rstrip("]\"'")]
    else:
        raw_blocks = [content]

    messages = []
    for block in raw_blocks:
        block = normalize_escapes(block)
        if not block.strip():
            continue
        for raw in _EMAIL_SPLIT_RE.split(block):
            raw = raw.strip()
            if not raw:
                continue
            head, _, body = raw.partition("\n\n")
            headers = dict(_EMAIL_HEADER_RE.findall(head))
            if not headers.get("From"):  # no header block: whole text is body
                headers, body = {}, raw
            messages.append({"headers": headers, "body": body.strip()})
    return messages


def gmail_to_items(doc: dict, database: str, collection: str) -> list[dict]:
    """One typed ``email`` item per message; ``thread_id`` = doc id, reply chain via ``reply_to_id``.

    (``reply_to_id`` is renamed to ``in_reply_to`` on the wire; see ``to_wire``.)
    """
    doc_id, title = doc["doc_id"], doc.get("title") or ""
    messages = parse_gmail_messages(doc.get("content") or "")
    items: list[dict] = []
    prev_ext: str | None = None
    for i, msg in enumerate(messages):
        h = msg["headers"]
        ext = f"{doc_id}_e{i:02d}" if len(messages) > 1 else doc_id
        subject = h.get("Subject") or title
        ts = _parse_gmail_date(h.get("Date", ""))
        fields: dict = {
            "kind": "email",
            "subject": _cap(subject, TITLE_MAX),
            "body": msg["body"],
            "thread_id": doc_id,
        }
        if h.get("From"):
            fields["from"] = h["From"]
        to = [t.strip() for t in (h.get("To") or "").split(",") if t.strip()]
        if to:
            fields["to"] = to
        cc = [t.strip() for t in (h.get("Cc") or "").split(",") if t.strip()]
        if cc:
            fields["cc"] = cc
        if prev_ext:
            fields["reply_to_id"] = prev_ext
        if ts:
            fields["created_at"] = ts
        items.append(_typed_item(
            item_id=ext, database=database, collection=collection,
            title=subject, kind="email", provider="gmail", external_id=ext, fields=fields,
            metadata={"erb_doc_id": doc_id, "erb_source_type": "gmail",
                      "thread_index": i, "thread_size": len(messages)},
            timestamp=ts,
        ))
        prev_ext = ext
    return items


# ---------------------------------------------------------------------------
# Tickets (jira / linear / github): description + optional "comments:" section
# ---------------------------------------------------------------------------
_COMMENTS_MARKER_RE = re.compile(
    r"^(?:comments|review_comments|review_conversation|customer_communication):\s*$",
    re.M)
# Comment-entry header, covering every observed corpus shape:
#   "Ava Chen: text"                                  (plain)
#   "2025-01-08 - Maya Chen: text"                    (dated)
#   "2025-01-11 - SRE Review (Ana Torres): text"      (dated + paren role)
#   "Jordan Rivers (reviewer): text"                  (paren role)
#   "Priya Shah (Support) 2026-03-12T16:18:00Z: text" (role + ISO timestamp)
_COMMENT_AUTHOR_RE = re.compile(
    r"^(?:(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2}(?::\d{2})?)?(?:\s*[-–—:]\s*|\s+))?"  # 1: date (+time)
    r"(\([^)]{1,40}\)|"                                  # 2: "(Author)" form, or
    r"[A-Z][A-Za-z .'’\-]{1,40}?"                        #    plain author name
    r"(?:\s*\([^)]{1,40}\))?)"                           #    with optional (role)
    r"\s*(\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+\-]\d{2}:?\d{2})?)?"  # 3: optional ISO ts
    r"\s*[:\-–—]\s(.*)$")
# A lowercase top-level section heading (merge_summary:, release_notes:, ...)
# terminates the comments section.
_SECTION_HEADING_RE = re.compile(r"^[a-z][a-z_ ]{2,40}:\s*$")


def _comment_author_ok(name: str, dated: bool) -> bool:
    """Accept dated entries, bot authors, paren-role authors and person-like names.

    Undated single capitalized words ("Question:", "Recommendation:") are
    colon-headers inside a comment body, not authors.
    """
    if dated or name.lower().endswith("-bot") or "(" in name:
        return True
    toks = name.split()
    return 2 <= len(toks) <= 4 and all(re.match(r"^[A-Z][A-Za-z'.\-]*$", t) for t in toks)


def split_ticket_comments(content: str) -> tuple[str, list[dict]]:
    """Split ticket content into ``(description, [{author, body, date}, ...])``."""
    m = _COMMENTS_MARKER_RE.search(content)
    if not m:
        return content, []
    description = content[: m.start()].rstrip()
    comments: list[dict] = []
    tail_lines: list[str] = []
    in_tail = False
    for line in content[m.end():].strip().splitlines():
        if in_tail:
            tail_lines.append(line)
            continue
        if _SECTION_HEADING_RE.match(line.strip()):
            # next top-level section: not comments, kept in the description
            in_tail = True
            tail_lines.append(line)
            continue
        cm = _COMMENT_AUTHOR_RE.match(line)
        if cm and _comment_author_ok(cm.group(2), dated=bool(cm.group(1) or cm.group(3))):
            date = cm.group(1)
            if not date and cm.group(3):  # ISO timestamp form
                date = cm.group(3)[:10]
            author = cm.group(2).strip().strip("()")
            comments.append({"author": author, "body": cm.group(4), "date": date})
        elif comments:
            comments[-1]["body"] += "\n" + line
        elif line.strip():  # text before any author line stays in the description
            description += "\n" + line
    if tail_lines:
        description += "\n\n" + "\n".join(tail_lines)
    return description, [c for c in comments if c["body"].strip()]


def ticket_to_items(doc: dict, database: str, collection: str) -> list[dict]:
    """A ``ticket`` item plus one ``comment`` item per comment (``fields.parent_id`` = ticket)."""
    st = doc["source_type"]
    _, provider = SOURCE_TYPE_TO_KIND_PROVIDER[st]
    doc_id, title = doc["doc_id"], doc.get("title") or ""
    content = normalize_escapes(doc.get("content") or "")
    description, comments = split_ticket_comments(content)

    base_meta = {"erb_doc_id": doc_id, "erb_source_type": st}
    items = [_typed_item(
        item_id=doc_id, database=database, collection=collection,
        title=title, kind="ticket", provider=provider, external_id=doc_id,
        fields={"kind": "ticket", "title": _cap(title, TITLE_MAX), "description": description},
        metadata={**base_meta, "comment_count": len(comments)},
    )]
    for i, c in enumerate(comments):
        ext = f"{doc_id}_c{i:02d}"
        fields = {"kind": "comment", "body": c["body"], "author": c["author"], "parent_id": doc_id}
        if c.get("date"):
            fields["created_at"] = f"{c['date']}T00:00:00Z"
        items.append(_typed_item(
            item_id=ext, database=database, collection=collection,
            title=f"Comment on: {_cap(title, 200)}", kind="comment", provider=provider,
            external_id=ext, fields=fields, metadata={**base_meta, "comment_index": i},
        ))
    return items


# ---------------------------------------------------------------------------
# Single-item documents (confluence / google_drive / fireflies / hubspot)
# ---------------------------------------------------------------------------
def single_item(doc: dict, database: str, collection: str) -> dict:
    """One typed item; the primary text goes into the kind's body field."""
    st = doc["source_type"]
    kind, provider = SOURCE_TYPE_TO_KIND_PROVIDER.get(st, ("custom", st))
    content = normalize_escapes(doc.get("content") or "")
    title = doc.get("title") or ""
    ext = doc["doc_id"]

    metadata: dict = {"erb_doc_id": ext, "erb_source_type": st}
    timestamp = None
    if st in ("confluence", "google_drive"):
        revisions, latest_iso = parse_revisions(content)
        if revisions:
            metadata["revisions"] = revisions
            metadata["revision_count"] = len(revisions)
            timestamp = latest_iso

    short = _cap(title, TITLE_MAX)
    if kind == "email":
        fields = {"kind": "email", "subject": short, "body": content}
    elif kind == "ticket":
        fields = {"kind": "ticket", "title": short, "description": content}
    elif kind == "knowledge_base":
        fields = {"kind": "knowledge_base", "title": short, "body": content}
    else:  # custom (hubspot): labelled sections become structured data keys
        data: dict = {"title": short}
        sections = re.split(r"(?m)^([a-z][a-z_ ]{2,40}):\s*$", content)
        if len(sections) >= 3:
            preamble = sections[0].strip()
            if preamble:
                data["body"] = preamble
            for key, text in zip(sections[1::2], sections[2::2], strict=False):
                data[key.strip().replace(" ", "_")] = text.strip()
        else:
            data["body"] = content
        fields = {"kind": "custom", "data": data}

    return _typed_item(
        item_id=ext, database=database, collection=collection, title=title,
        kind=kind, provider=provider, external_id=ext, fields=fields,
        metadata=metadata, timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Wire normalisation (what was actually POSTed)
# ---------------------------------------------------------------------------
def _flatten_metadata(meta: dict) -> dict:
    """Coerce metadata to the API's max-nesting-depth-1 contract."""

    def _scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def _stringify(v) -> str:
        if isinstance(v, dict):
            if all(_scalar(x) for x in v.values()):
                return ": ".join(str(x) for x in v.values() if x is not None)
            return json.dumps(v, ensure_ascii=False)
        return json.dumps(v, ensure_ascii=False) if not _scalar(v) else str(v)

    out: dict = {}
    for k, v in meta.items():
        if _scalar(v):
            out[k] = v
        elif isinstance(v, list):
            out[k] = [x if _scalar(x) else _stringify(x) for x in v]
        else:
            out[k] = _stringify(v)
    return out


def _normalize_reply_key(item: dict) -> None:
    """Rename the email reply pointer ``reply_to_id`` -> ``in_reply_to`` (the API's key)."""
    fields = item.get("fields")
    if isinstance(fields, dict) and fields.get("kind") == "email":
        rid = fields.pop("reply_to_id", None)
        if rid and not fields.get("in_reply_to"):
            fields["in_reply_to"] = rid


def to_wire(item: dict) -> dict:
    """Final per-item normalisation: drop ``None`` values, flatten metadata, rename the reply key."""
    out = {k: v for k, v in item.items() if v is not None}
    if isinstance(out.get("metadata"), dict):
        out["metadata"] = _flatten_metadata(out["metadata"])
    _normalize_reply_key(out)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def convert(doc: dict, database: str, collection: str) -> list[dict]:
    """Convert one corpus row (``doc_id``/``source_type``/``title``/``content``) into
    the typed ``app_knowledge`` items that are sent to HydraDB.

    Returns an empty list for the benchmark's handful of empty documents.
    """
    st = doc["source_type"]
    if not (doc.get("content") or "").strip():
        return []
    if st == "slack":
        items = slack_to_items(doc, database, collection)
    elif st == "gmail":
        items = gmail_to_items(doc, database, collection)
    elif st in ("jira", "linear", "github"):
        items = ticket_to_items(doc, database, collection)
    else:
        items = [single_item(doc, database, collection)]
    return [to_wire(it) for it in items]
