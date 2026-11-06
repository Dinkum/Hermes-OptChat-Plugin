"""Real Hermes metadata BLOBs must not prevent exact message recovery."""
import sqlite3

from optchat.config import Settings
from optchat.recovery import recover_archive
from optchat.store import Store


def test_recovery_accepts_binary_display_identity_without_altering_message(tmp_path):
    root = tmp_path
    home = root / "hermes"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as db:
        db.execute("""CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,
            content TEXT,timestamp REAL,message_uid TEXT,active INTEGER,compacted INTEGER,
            display_identity BLOB)""")
        db.execute("INSERT INTO messages VALUES (1,'session','user','Exact original',1,'message',1,0,?)",
                   (b'\x00\xff\x80display-key',))
    old = Store(root / "old","session",128000,Settings())
    try:
        old.sync_hermes(home,"session")
        old_path = old.root / "chat.db"
    finally:
        old.close()
    result = recover_archive(old_path,home / "state.db",root / "new","session")
    assert result["event_map"] == {0:0}
    recovered = Store(root / "new","session",128000,Settings())
    try:
        assert recovered.zoom(0,1)["text"] == "Exact original"
    finally:
        recovered.close()
