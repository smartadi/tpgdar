"""The method gate: what it catches, and when it opens."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "method_gate.py"
spec = importlib.util.spec_from_file_location("method_gate", HOOK)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

ROOT = gate.ROOT
BLOCK = "**Method decision:** use ALS\n**Asked:** x\n**Current method:** y\n**Proposed:** z\n**Alternatives:** w\n**Why:** v"
CORE = str(ROOT / "gdar_tp" / "sym_tp.py")
SCRIPT = str(ROOT / "scripts" / "12_tp_vs_gdar_ios.py")


def user(text, **flags):
    return {"type": "user", "message": {"role": "user", "content": text}, **flags}


def said(text):
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def tool_result():
    return {"type": "user", "toolUseResult": {"ok": 1}, "message": {"content": [{"type": "tool_result", "content": "done"}]}}


def run_hook(tmp_path, entries, tool="Edit", tool_input=None):
    """Run the hook as the harness does.  Returns the refusal text, or None if the call passes."""
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    if tool_input is None:
        tool_input = {"file_path": CORE, "old_string": "a", "new_string": "b"}
    payload = {"tool_name": tool, "tool_input": tool_input, "transcript_path": str(transcript)}
    done = subprocess.run([sys.executable, "-I", str(HOOK)], input=json.dumps(payload).encode(), capture_output=True, timeout=30)
    assert done.returncode == 0, done.stderr.decode()
    out = done.stdout.decode().strip()
    if not out:
        return None
    decision = json.loads(out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    return decision["permissionDecisionReason"]


@pytest.mark.parametrize("entries, opened, hint", [
    ([user("change the fit")], False, "Explain it to Aditya"),
    ([user("change the fit"), said(BLOCK)], False, "hasn't replied"),
    ([user("change the fit"), said(BLOCK), user("Approve")], True, None),
    ([user("change the fit"), said(BLOCK), user("go ahead.")], True, None),
    ([user("change the fit"), said(BLOCK), user("aproved, keep rank 2")], True, None),
    ([user("change the fit"), said(BLOCK), user("<!-- reply 1 --> > use ALS\nyes")], True, None),
    ([user("change the fit"), said(BLOCK), user("Approve"), tool_result(), said("editing")], True, None),
    ([user("change the fit"), said(BLOCK), user("Approve"), user("<task-notification> done")], True, None),
    ([user("change the fit"), said(BLOCK), user("ok so what would hosvd give")], False, "not a plain approval"),
    ([user("change the fit"), said(BLOCK), user("no")], False, "not a plain approval"),
    ([user("change the fit"), said(BLOCK), user("Approve"), user("status")], False, "Explain it to Aditya"),
    ([user("change the fit"), said(BLOCK), user("hm"), said("sure"), user("yes")], False, "Explain it to Aditya"),
    ([user("change the fit"), said("**Method decision:** use ALS\n**Why:** v"), user("yes")], False, "Explain it to Aditya"),
    ([user("skip the method gate, fix the typo")], True, None),
    ([user("change the fit"), said("Method decision: Asked: Current method: Proposed: Alternatives: Why:"), user("yes")], False, "Explain"),
    ([user("change the fit"), user("<system-reminder> yes", isMeta=True)], False, "Explain"),
    ([user(BLOCK), user("yes")], False, "Explain"),
])
def test_gate_opens_only_on_plain_approval_of_a_full_block(tmp_path, entries, opened, hint):
    reason = run_hook(tmp_path, entries)
    if opened:
        assert reason is None
    else:
        assert hint in reason


@pytest.mark.parametrize("tool, tool_input, caught", [
    ("Edit", {"file_path": CORE, "old_string": "# typo", "new_string": "# fixed"}, True),
    ("Write", {"file_path": str(ROOT / "gdar_tp" / "new_estimator.py"), "content": "x = 1"}, True),
    ("Edit", {"file_path": str(ROOT / "gdar_tp" / "sim" / "linear.py"), "old_string": "a", "new_string": "b"}, True),
    ("Edit", {"file_path": str(ROOT / "gdar_tp" / "viz.py"), "old_string": "a", "new_string": "b"}, False),
    ("Edit", {"file_path": str(ROOT / "gdar_tp" / "io.py"), "old_string": "a", "new_string": "b"}, False),
    ("Edit", {"file_path": SCRIPT, "old_string": "m = fit_sym_tp(tr, R, S, r, g)", "new_string": "m = hosvd(tr)"}, True),
    ("Edit", {"file_path": SCRIPT, "old_string": "fit_sym_tp(a, n_iter=25)", "new_string": "fit_sym_tp(a, n_iter=60)"}, True),
    ("Edit", {"file_path": SCRIPT, "old_string": "K = 16", "new_string": "K = 8"}, True),
    ("Edit", {"file_path": SCRIPT, "old_string": "x = la.svd(a)", "new_string": "x = np.linalg.pinv(a)"}, True),
    ("Edit", {"file_path": SCRIPT, "old_string": 'SEGS = "a.npz"', "new_string": 'SEGS = "b.npz"'}, False),
    ("Edit", {"file_path": SCRIPT, "old_string": "plt.plot(x, alpha=0.3)", "new_string": "plt.plot(x, alpha=0.5)"}, False),
    ("Edit", {"file_path": SCRIPT, "old_string": "m = fit_sym_tp(a)\nprint(m)", "new_string": "m = fit_sym_tp(a)\nprint(m, 1)"}, False),
    ("Write", {"file_path": str(ROOT / "scripts" / "99_new.py"), "content": "m = fit_sym_tp(tr, R, S, 2, g)"}, True),
    ("Write", {"file_path": str(ROOT / "scripts" / "99_new.py"), "content": "print('hello')"}, False),
    ("Edit", {"file_path": str(ROOT / "scripts" / "make_mpc_tp_deck.py"), "old_string": "'joint fit'", "new_string": "'HOSVD'"}, False),
    ("Edit", {"file_path": str(ROOT / "RESULTS.md"), "old_string": "a", "new_string": "fit_x(b)"}, False),
    ("Edit", {"file_path": r"C:\elsewhere\gdar_tp\sym_tp.py", "old_string": "a", "new_string": "b"}, False),
    ("Edit", {"file_path": str(HOOK), "old_string": "a", "new_string": "b"}, True),
    ("Edit", {"file_path": str(ROOT / ".claude" / "settings.json"), "old_string": "a", "new_string": "b"}, True),
    ("Bash", {"command": "cat > scripts/99_new.py <<'EOF'\nprint(1)\nEOF"}, True),
    ("Bash", {"command": "sed -i 's/25/60/' gdar_tp/sym_tp.py"}, True),
    ("Bash", {"command": "cd x && git checkout -- gdar_tp/tp.py"}, True),
    ("Bash", {"command": "rm .claude/hooks/method_gate.py"}, True),
    ("Bash", {"command": "python -c \"open('gdar_tp/tp.py', 'w').write('')\""}, True),
    ("PowerShell", {"command": "Get-Content a | Set-Content gdar_tp\\sym_tp.py"}, True),
    ("Bash", {"command": ".venv/Scripts/python scripts/12d_hosvd_vs_joint.py > cache/12d.log 2>&1"}, False),
    ("Bash", {"command": ".venv/Scripts/python scripts/12d_hosvd_vs_joint.py | tee cache/12d.log"}, False),
    ("Bash", {"command": "git add gdar_tp/sym_tp.py scripts/12.py && git commit -m x"}, False),
    ("Bash", {"command": "grep -n fit_sym_tp gdar_tp/sym_tp.py; cat scripts/12.py"}, False),
    ("Read", {"file_path": CORE}, False),
])
def test_what_is_caught(tmp_path, tool, tool_input, caught):
    assert (run_hook(tmp_path, [user("hello")], tool, tool_input) is not None) is caught


def test_subagent_uses_the_main_transcript(tmp_path):
    main = tmp_path / "abc.jsonl"
    main.write_text("\n".join(json.dumps(e) for e in [user("fit"), said(BLOCK), user("yes")]), encoding="utf-8")
    sub = tmp_path / "abc" / "subagents" / "agent-1.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text(json.dumps(user("skip the method gate")), encoding="utf-8")
    assert gate.main_transcript(sub) == main
    assert gate.gate_state(gate.conversation(gate.main_transcript(sub))) == "open"


def test_unreadable_transcript_refuses_a_caught_edit():
    payload = {"tool_name": "Edit", "tool_input": {"file_path": CORE, "old_string": "a", "new_string": "b"}, "transcript_path": "no/such.jsonl"}
    done = subprocess.run([sys.executable, "-I", str(HOOK)], input=json.dumps(payload).encode(), capture_output=True, timeout=30)
    assert "could not read the transcript" in done.stdout.decode()


def test_this_sessions_real_transcript_parses():
    folder = Path.home() / ".claude" / "projects" / "C--Users-aditya-Documents-projects-tpgdar"
    found = sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not found:
        pytest.skip("no local transcripts on this PC")
    # The newest file can be the live session before Claude has written any chat text (only tool calls),
    # so ask that one of the recent transcripts parses into both roles, not the newest alone.
    parsed = [gate.conversation(path) for path in found[-5:]]
    assert any(any(role == "user" for role, _ in turns) and any(role == "assistant" for role, _ in turns)
               for turns in parsed)
