"""A stricter cache policy must not silently reuse oversized immutable summaries."""
from dataclasses import replace
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from optchat.config import Settings
from optchat.store import Store
from optchat.worker import Worker
from test_identity_recovery import SESSION, engine_at, message, tool_result, workspace


def test_stricter_byte_policy_requires_new_generation_and_preserves_old_cache(tmp_path):
    root = tmp_path / 'archive'
    sentence_settings = Settings(chunk_limit='sentences')
    old = Store(root, 'main', 128000, sentence_settings)
    original = 'é' * 9000
    try:
        old.append('message:0', 'user', original)
        old.save_node(0, 1, original, {})
        old_path = old.root / 'chat.db'
    finally:
        old.close()
    before = hashlib.sha256(old_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match='migrat'):
        Store(root, 'main', 128000, replace(sentence_settings, chunk_limit='bytes'))
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == before
    with sqlite3.connect(old_path) as db:
        assert db.execute('SELECT text FROM nodes WHERE start=0 AND n=1').fetchone()[0] == original
        assert db.execute('SELECT text FROM events WHERE id=0').fetchone()[0] == original
        assert db.execute("SELECT value FROM meta WHERE key='chunk_limit'").fetchone()[0] == 'sentences'


def test_sparse_oversized_cover_migrates_without_losing_originals_or_replaying_history(tmp_path):
    from optchat.recovery import migrate_summary_limits
    settings = Settings(chunk_limit='sentences', view_bytes=4096, summary_tries=1)
    old = Store(tmp_path / 'old', 'main', 4096, settings)
    original = 'é' * 900
    try:
        for i in range(8):
            old.append(f'message:{i}', 'user', original if i % 2 == 0 else 'withdrawn')
            old.save_node(i, 1, original if i % 2 == 0 else 'withdrawn', {})
        with old.lock, old.db:
            old.reconcile_active([0, 2, 4, 6])
            old.db.execute('DELETE FROM jobs')
        assert len(old.render().encode()) > 4096
        source = old.root / 'chat.db'
        expected_parts = old.parts()
    finally:
        old.close()
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []

    def summarize(messages):
        calls.append(messages)
        return SimpleNamespace(text='user: Historical multilingual instruction; retrieve its original for details.',
                               model='fixture', provider='none', usage=SimpleNamespace())

    # Worker usage is a dataclass supplied by the host adapter.
    from optchat.engine import Usage
    def measured_summary(messages):
        result = summarize(messages)
        result.usage = Usage()
        return result

    strict = replace(settings, chunk_limit='bytes')
    report = migrate_summary_limits(source, tmp_path / 'new', settings=strict,
                                    budget=4096, summarize=measured_summary)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert len(calls) == 4, 'Only oversized frontier nodes, not every original, need model calls'
    new = Store(tmp_path / 'new', 'main', 4096, strict)
    try:
        assert new.parts() == expected_parts
        assert len(new.render().encode()) <= 2048
        assert new.count() == 8
        assert new.zoom(0, 1)['text'] == original
        assert new.zoom(1, 1)['text'] == 'withdrawn'
        assert all(len(new.node(start, n).encode()) <= 512 for start, n in new.parts())
        assert report['rebuilt_frontier_nodes'] == 4
        assert new.ready_job() is None
    finally:
        new.close()
    resumed = Store(tmp_path / 'new', 'main', 4096, strict)
    try:
        assert resumed.parts() == expected_parts
        assert resumed.ready_job() is None
        assert resumed.zoom(6, 1)['text'] == original
    finally:
        resumed.close()


@pytest.mark.parametrize("failure", ["provider_error", "empty_summary"])
def test_failed_migration_never_publishes_a_generation(tmp_path, failure):
    from optchat.recovery import migrate_summary_limits
    from optchat.engine import Usage
    old = Store(tmp_path / 'old', 'main', 128000, Settings(chunk_limit='sentences'))
    try:
        old.append('message:0', 'user', 'Important original. ' * 100)
        old.save_node(0, 1, 'é' * 900, {})
        source = old.root / 'chat.db'
    finally:
        old.close()
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    def summarize(messages):
        if failure == 'provider_error':
            raise ConnectionError('Unavailable test provider')
        return SimpleNamespace(text='', model='fixture', provider='none', usage=Usage())

    with pytest.raises((ConnectionError, RuntimeError)):
        migrate_summary_limits(source, tmp_path / 'new',
                               settings=Settings(summary_tries=1), summarize=summarize)
    assert not (tmp_path / 'new' / source.parent.name).exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_migration_keeps_older_audit_generations_retrievable_without_reviving_them(workspace, engine_at):
    from optchat.recovery import migrate_summary_limits
    from optchat.engine import Usage
    import json
    root, home, database = workspace
    original = 'é' * 900
    message(database, 1, 'current', original)
    elder = Store(root / 'elder', SESSION, 128000, Settings())
    elder.append('retired', 'user', 'Retired audit-only instruction')
    elder_path = elder.root / 'chat.db'
    elder.close()
    old = Store(root / 'old', SESSION, 128000, Settings(chunk_limit='sentences'))
    try:
        old.sync_hermes(home, SESSION)
        old.save_node(0, 1, original, {})
        with old.db:
            old.db.execute("INSERT INTO meta VALUES ('recovery_report',?)",
                           (json.dumps({'source_archive': str(elder_path)}),))
        source = old.root / 'chat.db'
    finally:
        old.close()
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, elder_path)}

    def summarize(messages):
        return SimpleNamespace(text='user: Preserve the current instruction.', model='fixture',
                               provider='none', usage=Usage())

    strict = Settings(summary_tries=1)
    migrate_summary_limits(source, home / 'optchat', settings=strict, summarize=summarize)
    engine = engine_at(strict)
    assert tool_result(engine, 'optchat_zoom', {'id': 0, 'n': 1})['text'] == original
    assert tool_result(engine, 'optchat_zoom', {'id': 0, 'n': 1, 'archive': 'previous'})['text'] == original
    assert tool_result(engine, 'optchat_zoom', {
        'id': 0, 'n': 1, 'archive': 'previous', 'generation': 2,
    })['text'] == 'Retired audit-only instruction'
    assert tool_result(engine, 'optchat_search', {'query': 'Retired audit-only'})['matches'] == []
    for path, checksum in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    # Reusing the original source later cannot mutate the captured generation.
    late = Store(root / 'old', SESSION, 128000, Settings(chunk_limit='sentences'))
    try:
        late.append('post-migration', 'user', 'Post-migration addition must remain outside the snapshot')
    finally:
        late.close()
    assert tool_result(engine, 'optchat_search', {
        'query': 'Post-migration addition', 'archive': 'previous',
    })['matches'] == []


def test_recipe_view_batches_to_half_and_restores_the_saved_cover(tmp_path):
    settings = Settings()
    store = Store(tmp_path / 'memory', 'main', 128000, settings)
    previous_bytes, batches = 0, []
    try:
        for i in range(280):
            store.append(f'message:{i}', 'user', f'Original message {i}')
            store.save_node(i, 1, f'Message {i}: ' + 'x' * 480, {})
            for start,n,_ in store.parents(i):
                store.save_node(start, n, f'Messages {start} through {start+n-1}: ' + 'x' * 470, {})
            current_bytes = len(store.render().encode())
            assert current_bytes <= 128000
            if previous_bytes-current_bytes > 50000:
                batches.append(current_bytes)
            previous_bytes = current_bytes
        assert batches and all(value <= 64000 for value in batches)
        saved_view, saved_parts = store.render(), store.parts()
    finally:
        store.close()
    resumed = Store(tmp_path / 'memory', 'main', 128000, settings)
    try:
        assert resumed.render() == saved_view
        assert resumed.parts() == saved_parts
        covered = [i for start,n in resumed.parts() for i in range(start,start+n)]
        assert covered == list(range(280))
    finally:
        resumed.close()


def test_withdrawal_after_migration_rebuilds_exposed_descendants_from_originals(tmp_path):
    from optchat.recovery import migrate_summary_limits
    from optchat.engine import Usage
    settings = Settings(chunk_limit='sentences')
    old = Store(tmp_path / 'old', 'main', 128000, settings)
    originals = [f'Original instruction {i}: ' + 'é' * 900 for i in range(8)]
    try:
        for i,original in enumerate(originals):
            old.append(f'message:{i}', 'user', original)
            old.save_node(i, 1, original, {})
            for start,n,_ in old.parents(i):
                old.save_node(start, n, 'Legacy whole-range memory: ' + 'é' * 900, {})
        with old.db:
            old.fit(limit=1, next_message=True)
        assert old.parts() == [(0, 8)]
        source = old.root / 'chat.db'
    finally:
        old.close()

    def summarize(messages):
        return SimpleNamespace(text='user: Preserve historical instructions; zoom for exact details.',
                               model='fixture', provider='none', usage=Usage())

    strict = Settings(summary_tries=1, summary_concurrency=1, merge_lanes=1,
                      settle_seconds=3, summary_timeout=1, retry_seconds=0.01)
    migrate_summary_limits(source, tmp_path / 'new', settings=strict, summarize=summarize)
    store = Store(tmp_path / 'new', 'main', 128000, strict)
    with store.lock, store.db:
        store.reconcile_active(range(7))
    worker = Worker(store, strict, summarize)
    try:
        view = worker.settle(store.count())
        assert '0+8|' not in view
        assert len(view.encode()) <= 128000
        assert [i for start,n in store.parts() for i in range(start,start+n)] == list(range(7))
        assert store.zoom(7, 1)['text'] == originals[7]
        assert all(store.node(start,n) is not None for start,n in store.parts())
    finally:
        worker.close()
