from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.reply_compose import compose_reply, _trim
from spotify_agent.retrieve import RetrievedExample


def test_compose_reply_account_security():
    reply = compose_reply("account_security", "Someone changed my email", [], "escalate")
    assert "unauthorized access" in reply
    assert "no passwords" in reply
    assert "DM" in reply


def test_compose_reply_billing_variants():
    # General billing
    general = compose_reply("billing_charge", "Charged twice this month", [], "escalate")
    assert "screenshot of the charge" in general
    assert "hide full card digits" in general

    # Refund variant
    refund = compose_reply("billing_charge", "Where is my refund?", [], "escalate")
    assert "refund delay" in refund

    # Payment update variant
    payment = compose_reply("billing_charge", "Need to update my card info", [], "escalate")
    assert "update your payment method" in payment


def test_compose_reply_playback_bug():
    reply = compose_reply("playback_bug", "Songs keep pausing", [], "escalate")
    assert "device + OS + Spotify app version" in reply


def test_compose_reply_feature_howto_with_history():
    example = RetrievedExample(
        pair_id="pair-1",
        customer_text="How do I make a collaborative playlist?",
        brand_reply_text="Go to your playlist, tap the three dots, and select Invite collaborators for your device.",
        score=0.9,
    )
    reply = compose_reply("feature_howto", "How do I make collaborative playlist?", [example], "auto")
    assert "Invite collaborators" in reply


def test_compose_reply_rejects_unsafe_history():
    unsafe_example = RetrievedExample(
        pair_id="pair-2",
        customer_text="Login help",
        brand_reply_text="Please reply with your password and email address.",
        score=0.95,
    )
    reply = compose_reply("login_access", "Cannot login", [unsafe_example], "escalate")
    assert "your password and email address" not in reply
    assert "don't send passwords" in reply


def test_compose_reply_none_customer_text():
    reply = compose_reply("cancel_premium", None, [], "escalate")
    assert "cancelling/stopping Premium" in reply


def test_trim_whitespace_and_length():
    padded = "   Hello    world   with   spaces   "
    assert _trim(padded) == "Hello world with spaces"

    long_text = "A" * 500
    trimmed = _trim(long_text, limit=100)
    assert len(trimmed) == 100
    assert trimmed.endswith("…")
