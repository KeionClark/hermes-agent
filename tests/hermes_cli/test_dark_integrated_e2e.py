"""One-off disposable-board witness for exact dark publisher + router integration."""
import json
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_db import create_task, claim_task, block_task, unblock_task, complete_task
from hermes_cli.kanban_event_router import Subscriber, enqueue_inbox, run_once


def test_combined_disposable_lifecycle_and_old_task_events(tmp_path, monkeypatch):
    path = tmp_path / 'board.db'
    monkeypatch.setenv('HERMES_KANBAN_DB', str(path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes_home'))
    conn = connect(path)
    conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES('old','blocked','{}',1)")
    old = tuple(conn.execute("SELECT id,task_id,kind,payload,created_at FROM task_events WHERE task_id='old'").fetchone())
    regs = {'all': Subscriber(frozenset({'card.created','card.status_changed','card.blocked','card.completed'}), enqueue_inbox('all')),
            'blocked': Subscriber(frozenset({'card.blocked'}), enqueue_inbox('blocked'))}
    assert run_once(conn, regs) == {'all': 0, 'blocked': 0}
    task_id = create_task(conn, title='dark-only', assignee='synthetic')
    claimed = claim_task(conn, task_id)
    assert claimed is not None
    assert block_task(conn, task_id, reason='synthetic', kind='needs_input', expected_run_id=claimed.current_run_id)
    assert unblock_task(conn, task_id)
    assert complete_task(conn, task_id, result='synthetic', fire_lifecycle_hook=False)
    events = [(r[0], r[1], json.loads(r[2])) for r in conn.execute('SELECT id,event_type,payload FROM hermes_events ORDER BY id') if json.loads(r[2]).get('task_id') == task_id]
    types = [kind for _, kind, _ in events]
    assert types.count('card.created') == types.count('card.blocked') == types.count('card.completed') == 1, types
    assert types.count('card.status_changed') >= 4, types
    assert all(set(payload) <= {'task_id','from','to','run_id','block_kind'} for _, _, payload in events)
    assert run_once(conn, regs) == {'all': len(events), 'blocked': 1}
    rows = list(conn.execute('SELECT subscriber_id,event_id,event_type FROM hermes_router_inbox ORDER BY subscriber_id,event_id'))
    assert len(rows) == len(events) + 1
    assert tuple(conn.execute("SELECT id,task_id,kind,payload,created_at FROM task_events WHERE task_id='old'").fetchone()) == old
    assert conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == 'done'
    conn.close()
    conn = connect(path)
    assert run_once(conn, regs) == {'all': 0, 'blocked': 0}
    assert conn.execute('SELECT COUNT(*) FROM hermes_router_inbox').fetchone()[0] == len(rows)
    conn.close()
