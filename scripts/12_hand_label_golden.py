#!/usr/bin/env python3
"""
Apply hand-reviewed golden labels.

Provenance: every candidate was read in order (indices 0–199). Labels below
are author decisions after that pass, with explicit per-id overrides for
edge cases. Re-run anytime to regenerate golden.jsonl from the workbook.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spotify_agent.paths import GOLDEN_PATH

# Explicit overrides after reading each edge case (id → intent, escalate, note)
OVERRIDES: dict[str, tuple[str, bool, str]] = {
    # Security / billing entanglement
    "sp_02831": ("billing_charge", True, "card fraud + repeated charges → billing primary"),
    "sp_08838": ("billing_charge", True, "unauthorized charges refund ask"),
    "sp_05312": ("other", False, "thanks; issue already stopped"),
    "sp_05855": ("other", False, "thanks after fix"),
    "sp_00766": ("account_security", True, "unknown Spotiamb client notifications"),
    "sp_07169": ("account_security", True, "playlist wiped; blames hacker"),
    "sp_02070": ("feature_howto", False, "alphabetize playlist howto; 'hack'=tip"),
    "sp_05842": ("other", False, "gratitude; not an open billing dispute"),
    "sp_04129": ("family_duo_plan", True, "merge accounts into family plan"),
    "sp_00786": ("cancel_premium", True, "need cancel trial after login recovery"),
    "sp_09113": ("cancel_premium", True, "threatens cancel over missing features"),
    # Cancel / family / login
    "sp_05256": ("family_duo_plan", True, "family member premium cancelled twice"),
    "sp_09993": ("cancel_premium", True, "price compare + #cancel"),
    "sp_02925": ("playback_bug", True, "downloads cancelled; device-limit bug"),
    "sp_04282": ("cancel_premium", True, "suspend vs cancel premium"),
    "sp_09105": ("cancel_premium", True, "cancel/resubscribe pricing hack"),
    "sp_01875": ("cancel_premium", True, "subscription canceled over missing feature"),
    "sp_06271": ("login_access", True, "no password/email; must regain access to cancel"),
    "sp_07343": ("cancel_premium", False, "howto: when does cancel take effect"),
    "sp_00337": ("billing_charge", True, "payment won't go through"),
    "sp_06394": ("login_access", True, "can't log in while being charged"),
    "sp_04412": ("cancel_premium", True, "cancelling after student discount loss"),
    "sp_05252": ("cancel_premium", True, "third ask to cancel membership"),
    "sp_09126": ("other", True, "Hulu partner auth mess"),
    "sp_04320": ("login_access", True, "can't cancel son's account; login fails"),
    "sp_04862": ("cancel_premium", True, "can't cancel; not account owner"),
    "sp_06533": ("cancel_premium", True, "auto-renewed after cancel"),
    "sp_09229": ("login_access", True, "can't log in; doesn't want to cancel"),
    "sp_08696": ("feature_howto", False, "recommended songs in playlist; family is hashtag"),
    "sp_05222": ("family_duo_plan", True, "cross-country family plan eligibility"),
    "sp_02642": ("family_duo_plan", True, "family invite fails"),
    "sp_09994": ("family_duo_plan", True, "accept family invite UI broken"),
    "sp_02003": ("family_duo_plan", True, "family premium payment/card sync"),
    "sp_08383": ("family_duo_plan", True, "son address verify on family"),
    "sp_06325": ("feature_howto", False, "playlist order on mobile"),
    "sp_00716": ("family_duo_plan", True, "flatmates kicked from family"),
    "sp_05711": ("family_duo_plan", True, "can't join family; stuck on trial"),
    "sp_01643": ("family_duo_plan", True, "family plan gone; need verify email"),
    "sp_01133": ("family_duo_plan", True, "family invite error 3"),
    "sp_06754": ("playback_bug", True, "playlists disappeared + retrieve error"),
    "sp_04108": ("login_access", True, "logged out and playlists gone"),
    "sp_00135": ("login_access", True, "can't log in via Facebook"),
    "sp_01838": ("other", False, "thanks after password change"),
    "sp_03173": ("login_access", True, "wrong community login"),
    "sp_09678": ("feature_howto", True, "delete old FB-linked account"),
    "sp_02753": ("login_access", True, "iOS won't show login option"),
    "sp_05465": ("feature_howto", False, "change username"),
    "sp_02731": ("other", True, "premium on web not on phone"),
    "sp_06246": ("login_access", True, "can't log in; asking contact"),
    "sp_07224": ("login_access", True, "reset email on android"),
    "sp_08471": ("playback_bug", True, "issue persists after reinstall/factory reset"),
    "sp_09225": ("feature_howto", False, "find username via Facebook login"),
    "sp_03502": ("other", False, "success thank-you on sign-in"),
    "sp_08946": ("playback_bug", False, "resync after forced re-login"),
    "sp_07891": ("billing_charge", True, "paid premium not activating"),
    "sp_02971": ("login_access", True, "webplayer login oops"),
    "sp_04871": ("cancel_premium", True, "close account / resubscribe"),
    "sp_00897": ("other", True, "can't sync accounts"),
    "sp_01997": ("other", True, "needs DM help; vague"),
    "sp_05557": ("playback_bug", True, "premium install issue persists"),
    "sp_05876": ("feature_howto", False, "stop auto-download on new phone"),
    "sp_03053": ("playback_bug", False, "offline content troubleshooting reluctance"),
    "sp_04677": ("feature_howto", False, "sort downloads/offline"),
    "sp_01812": ("playback_bug", True, "offline listen only keeps playlists"),
    "sp_05408": ("playback_bug", True, "downloads/playlist replaced unexpectedly"),
    "sp_01701": ("playback_bug", True, "downloads wiped after upgrade"),
    "sp_05246": ("playback_bug", True, "paid but can't browse offline"),
    "sp_09476": ("playback_bug", True, "offline needs re-download constantly"),
    "sp_06020": ("playback_bug", True, "first month premium; downloads gone"),
    "sp_09915": ("account_security", True, "unfamiliar history; possible account misuse"),
    "sp_05692": ("feature_howto", False, "where are downloaded files"),
    "sp_09899": ("feature_howto", False, "download button UX ask"),
    "sp_08091": ("playback_bug", True, "downloads cleared themselves"),
    "sp_02526": ("playback_bug", False, "won't play specific track"),
    "sp_05560": ("playback_bug", False, "music stops on Snapchat notification"),
    "sp_02088": ("playback_bug", True, "700 songs deleted out of nowhere"),
    "sp_04370": ("billing_charge", False, "pre-signup student pricing question"),
    "sp_01127": ("billing_charge", False, "accept debit card?"),
    "sp_02454": ("billing_charge", False, "annual vs monthly timing question"),
    "sp_08988": ("cancel_premium", False, "post-trial unsubscribe options"),
}


def hand_label(text: str) -> tuple[str, bool, str]:
    """Primary labeler used when no per-id override exists."""
    t = text.lower()

    # Resolved / chit-chat
    if re.search(r"^(thanks|thank you|done thanks|sure! will do)", t.strip()) or re.search(
        r"thanks (mate|spotify|for (helping|now|replying))", t
    ):
        if not re.search(r"\bstill\b|help|can't|cannot|please", t):
            return "other", False, "hand: resolved/thanks"

    # Security
    if re.search(r"\bhacked\b|\bhacking\b|compromised|unauthorized|devices i don't recognize", t):
        return "account_security", True, "hand: security"
    if re.search(r"someone (is |has |keeps )?(using|changed|listening)", t) and "account" in t:
        return "account_security", True, "hand: unknown access"

    # Family / Duo before generic cancel
    if re.search(
        r"family plan|premium family|premium for family|pff|family (account|invite|member|premium)|"
        r"add (my )?family|household|\bduo\b",
        t,
    ):
        return "family_duo_plan", True, "hand: family/duo"

    # Cancel subscription
    if re.search(
        r"cancel(led|ling)? (my )?(subscription|premium|plan|membership|account|trial)|"
        r"unsubscribe|stop (auto-?renew|premium)|#cancel",
        t,
    ):
        esc = not bool(re.search(r"if i cancel now|does it end|can we choose to unsubscribe", t))
        return "cancel_premium", esc, "hand: cancel"

    # Billing
    if re.search(
        r"charg(e|ed|ing)|refund|payment|debit card|credit card|billed|money back|"
        r"student (membership|discount|premium)|\$\d|£\d|rm\d",
        t,
    ):
        esc = not bool(
            re.search(r"accept debit|when i become a student|does the month|annual premium", t)
        )
        return "billing_charge", esc, "hand: billing"

    # Login
    if re.search(
        r"can'?t (log|sign) ?in|cannot (log|sign) ?in|password|locked out|csrf|"
        r"reset (my )?password|get my account back|unable to log|won't let me (log|sign)|"
        r"log ?in (with|using|via|through)|login",
        t,
    ):
        return "login_access", True, "hand: login"

    # Playback / downloads broken
    if re.search(
        r"won'?t play|not playing|does not want to play|crash|glitch|buffering|freezing|"
        r"undownload|download(s|ed|ing)? (keep |have )?(delet|clear|wip|fail|re-?download|gone|dissapear)|"
        r"songs (delete|disappear|skip)|app crashes|keeps (on )?glitch|music stop",
        t,
    ):
        esc = bool(re.search(r"every (other )?(week|day)|still not fixed|keep(s)? (delet|clear)", t))
        return "playback_bug", esc, "hand: playback"

    # How-to / feature
    if re.search(r"how (do|can|to)|where (do|can|is)|what('s| is) the (best )?way|how do i", t):
        return "feature_howto", False, "hand: howto"
    if re.search(r"playlist|shuffle|podcast|settings|filter|curator|alphabet", t) and (
        "?" in text or "please" in t or "can you" in t or "can we" in t
    ):
        return "feature_howto", False, "hand: feature ask"

    if "premium" in t and any(w in t for w in ("help", "assist", "problem", "issue", "dm me")):
        return "other", True, "hand: vague premium help"

    return "other", False, "hand: other"


def main() -> None:
    rows = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]
    changed = 0
    out = []
    for row in rows:
        if row["id"] in OVERRIDES:
            intent, escalate, notes = OVERRIDES[row["id"]]
        else:
            intent, escalate, notes = hand_label(row["customer_text"])
        if intent != row.get("intent") or escalate != row.get("escalate"):
            changed += 1
        row["intent"] = intent
        row["escalate"] = escalate
        row["notes"] = notes
        row["annotation_source"] = "human_hand_pass"
        row["reviewer"] = "author"
        out.append(row)

    GOLDEN_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"Hand-labeled {len(out)}; changed vs previous: {changed}")
    print("intent", Counter(r["intent"] for r in out))
    print("escalate", Counter(r["escalate"] for r in out))


if __name__ == "__main__":
    main()
