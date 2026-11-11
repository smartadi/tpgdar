r"""Method gate: Claude may not change how a model is fitted or scored until Aditya has read why and approved it.

PreToolUse hook, wired up in .claude/settings.json.  It reads the tool call, and for a call that the rules below
catch it reads the session transcript, which Claude cannot write a user message into.  The sequence it enforces:

  1. Claude tries the edit and is refused.
  2. Claude writes the decision block (BLOCK_FIELDS) in chat and ends its turn.
  3. Aditya's next message is a plain approval (PLAIN_APPROVALS).
  4. Edits go through until Aditya's next message, which closes the gate again.

What it catches, in project-relative paths:
  - any edit under gdar_tp/ except FREE_MODULES
  - an edit to scripts/*.py that adds, removes or changes a fitting or scoring term (METHOD_TERMS)
  - this file and the settings that load it
  - a shell command that visibly writes to any of those; it does not see a write through a variable path

What it cannot see: the intent of an edit, so a comment fix in gdar_tp/ is stopped too, and a method choice made in
a scratchpad script is not.

A message containing "skip the method gate" opens the gate for that one message, for when the block isn't worth it.
"""

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREE_MODULES = {"gdar_tp/__init__.py", "gdar_tp/io.py", "gdar_tp/viz.py"}
SELF_FILES = {".claude/hooks/method_gate.py", ".claude/settings.json", ".claude/settings.local.json"}

METHOD_TERMS = re.compile(
    r"""\bfit_\w+\( | \bhosvd\( | \bhooi\( | \bsvd\( | \blstsq\( | \b_solve\( | \b_quad\(
      | \bpolytope_vertices\( | \bconvex_hull\( | \bsymmetric_basis\(
      | \blinalg\.(?:svd|lstsq|solve|pinv|inv|cholesky|qr|eigh?)\( | \bcvxpy\b | \bcp\.(?:Problem|Variable|Minimize)\b
      | \b(?:ridge|lam|lambda_|jitter|JITTER|n_iter|tol|starts|K|P|RANKS|NFOLD|g0|rank)\s*=\s*[^,\s)]+
      | \b(?:remove_evoked|evoked_design|sigma_of|folds|fold_stats|ll|train_loss|score)\(
      | \bloglik\w*\( | \bdef\s+\w*(?:fit|loss|score|likelihood)\w*
    """,
    re.X,
)

BLOCK_FIELDS = ("Method decision", "Asked", "Current method", "Proposed", "Alternatives", "Why")
BLOCK_TEMPLATE = """\
Method decision: the call in one line
Asked: what Aditya asked for, in their words
Current method: what the code does now, and what it gives on their data
Proposed: what you would change, with the math in a line or two
Alternatives: the others you weighed, keeping the current method among them
Why: why this one, with the evidence, or say there is none yet"""

PLAIN_APPROVALS = {
    "yes", "y", "yep", "yeah", "yes please", "yes do it", "yes go ahead", "go", "go ahead", "go for it", "do it", "ok",
    "okay", "ok do it", "ok go", "ok go ahead", "okay do it", "okay go ahead", "proceed", "sure", "sounds good",
    "looks good", "lgtm",
}
SKIP_PHRASE = re.compile(r"\bskip (?:the )?method gate\b", re.I)
# Messages the app delivers in the user role.  Anything else counts as Aditya speaking, which can only close the gate.
NOT_FROM_USER = ("<task-notification", "<side-sessions-event", "<ci-monitor-event", "<system-reminder", "<command-",
                 "<local-command", "<cross-session", "<teammate", "<channel")

GATED_IN_COMMAND = r"(?:gdar_tp[\\/][\w./\\-]+\.py|scripts[\\/][\w.-]+\.py|\.claude[\\/](?:hooks[\\/]method_gate\.py|settings(?:\.local)?\.json))"
SHELL_WRITES = (
    re.compile(r"(?:>>?|\btee(?:\s+-a)?)\s*[\"']?[^\s\"'|;&]*" + GATED_IN_COMMAND, re.I),
    re.compile(r"(?:sed\s+-i|\bperl\s+-pi|Set-Content|Add-Content|Out-File|\bpatch\b|\bcp\b|\bmv\b|\brm\b|Copy-Item|Move-Item|Remove-Item"
               r"|git\s+(?:apply|restore|checkout))[^|;&\n]*" + GATED_IN_COMMAND, re.I),
    re.compile(GATED_IN_COMMAND + r"[^|;&\n]*(?:Set-Content|Add-Content|Out-File|write_text|write_bytes)", re.I),
    re.compile(r"open\([^)]*" + GATED_IN_COMMAND + r"[^)]*[\"'](?:w|a|x|r\+)b?\+?[\"']", re.I),
    re.compile(r"(?:write_text|write_bytes|\.write\()[^\n]*" + GATED_IN_COMMAND, re.I),
)


def project_path(file_path, root=ROOT):
    """Project-relative path with forward slashes, lower-cased, or None if outside the project."""
    if not file_path:
        return None
    path = Path(file_path)
    if not path.is_absolute():
        path = root / path
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return None
    return rel.as_posix().lower()


def term_changes(old, new):
    """The fitting or scoring terms that an edit adds or removes."""
    before, after = (collections.Counter(METHOD_TERMS.findall(text or "")) for text in (old, new))
    return sorted((before - after) + (after - before))


def edit_texts(tool, tool_input, root=ROOT):
    """Old and new text of a file edit, as far as the tool call shows them."""
    if tool == "Edit":
        return tool_input.get("old_string", ""), tool_input.get("new_string", "")
    if tool == "MultiEdit":
        edits = tool_input.get("edits") or []
        return "\n".join(e.get("old_string", "") for e in edits), "\n".join(e.get("new_string", "") for e in edits)
    if tool == "Write":
        path = Path(tool_input.get("file_path", ""))
        path = path if path.is_absolute() else root / path
        try:
            old = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            old = ""
        return old, tool_input.get("content", "")
    return "", ""


def why_gated(tool, tool_input, root=ROOT):
    """A short reason if this call is caught, otherwise None."""
    if tool in ("Bash", "PowerShell"):
        command = tool_input.get("command", "")
        if any(rule.search(command) for rule in SHELL_WRITES):
            return "this shell command writes to fitting or scoring code, or to the gate"
        return None
    if tool not in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return None
    rel = project_path(tool_input.get("file_path") or tool_input.get("notebook_path"), root)
    if rel is None:
        return None
    if rel in SELF_FILES:
        return f"{rel} is the gate itself"
    if rel.startswith("gdar_tp/") and rel not in FREE_MODULES:
        return f"{rel} holds the model code"
    if rel.startswith("scripts/") and rel.endswith(".py"):
        changes = term_changes(*edit_texts(tool, tool_input, root))
        if changes:
            return f"this edit to {rel} changes fitting or scoring terms: {', '.join(changes[:6])}"
    return None


def main_transcript(path):
    """A subagent's transcript lives under <session>/subagents/; approval only ever comes from the main one."""
    path = Path(path)
    if "subagents" in path.parts:
        cut = path.parts.index("subagents")
        path = Path(*path.parts[:cut]).with_suffix(".jsonl")
    return path


def conversation(path):
    """Real user messages and Claude's chat text, in order, as (role, text)."""
    turns = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"type":"user"' not in line.replace(" ", "") and '"type":"assistant"' not in line.replace(" ", ""):
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            role = entry.get("type")
            if role not in ("user", "assistant") or entry.get("isSidechain"):
                continue
            content = (entry.get("message") or {}).get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            else:
                continue
            if not text.strip():
                continue
            if role == "user":
                if entry.get("isMeta") or entry.get("isCompactSummary") or entry.get("toolUseResult") is not None:
                    continue
                if text.lstrip().startswith(NOT_FROM_USER):
                    continue
            turns.append((role, text))
    return turns


def has_full_block(text):
    return all(re.search(rf"(?im)^[\s>*_#-]*{name}[*_]*\s*:", text) for name in BLOCK_FIELDS)


def is_plain_approval(text):
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = " ".join(text.split())
    return text in PLAIN_APPROVALS or re.match(r"(?:i )?ap+rove?d?\b", text) is not None


def gate_state(turns):
    """'open', or why it is shut: 'no block', 'waiting' for a reply to the block, 'not approved'."""
    users = [i for i, (role, _) in enumerate(turns) if role == "user"]
    last = users[-1] if users else -1
    if last >= 0 and SKIP_PHRASE.search(turns[last][1]):
        return "open"
    if any(role == "assistant" and has_full_block(text) for role, text in turns[last + 1:]):
        return "waiting"
    previous = users[-2] if len(users) > 1 else -1
    if last < 0 or not any(role == "assistant" and has_full_block(text) for role, text in turns[previous + 1:last]):
        return "no block"
    return "open" if is_plain_approval(turns[last][1]) else "not approved"


def refusal(reason, state):
    how = {
        "no block": "Explain it to Aditya before you touch it. In chat, write this block, then end your turn:\n\n"
                    + BLOCK_TEMPLATE + "\n\nIf you think this is not a method change, say so in the block. Aditya still decides.",
        "waiting": "You have written the block and Aditya hasn't replied. End your turn and wait.",
        "not approved": "Aditya's last message was not a plain approval. Deal with what they said, and if the edit is still "
                        "wanted, write the block again, updated, and end your turn:\n\n" + BLOCK_TEMPLATE,
    }[state]
    return (f"Method gate: {reason}. Nothing was changed.\n{how}\n\n"
            "The edit goes through once Aditya's next message is a plain approval (approve, yes, go ahead). "
            "Any later message from them closes the gate again. Do not work around this.")


def deny(text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                             "permissionDecisionReason": text}}))


def main():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
    reason = why_gated(payload.get("tool_name", ""), payload.get("tool_input") or {})
    if reason is None:
        return
    try:
        state = gate_state(conversation(main_transcript(payload["transcript_path"])))
    except Exception as exc:  # a caught edit stays refused if the transcript can't be read
        deny(f"Method gate: {reason}, and the gate could not read the transcript ({exc!r}). Nothing was changed. Tell Aditya.")
        return
    if state != "open":
        deny(refusal(reason, state))


if __name__ == "__main__":
    main()
