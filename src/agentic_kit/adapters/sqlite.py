"""SQLite-backed conversation and run stores."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from typing import TypeVar

from ..domain.models import ConversationState, RunRecord
from ..ports.stores import ConversationStore, RunStore

T = TypeVar("T")


class _Database:
    """One connection and one lock shared by both stores."""

    def __init__(self, path: str) -> None:
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = asyncio.Lock()
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                agent_id TEXT,
                sop_id TEXT,
                kind TEXT NOT NULL,
                event TEXT,
                status TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS runs_conversation_created
                ON runs (conversation_id, created_at);
            """
        )
        self.connection.commit()

    async def call(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        async with self.lock:
            return await asyncio.to_thread(operation, self.connection)


class SqliteConversationStore:
    def __init__(self, database: _Database) -> None:
        self._database = database

    async def get(self, conversation_id: str) -> ConversationState | None:
        def read(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                "SELECT payload FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return row["payload"] if row else None

        payload = await self._database.call(read)
        return ConversationState.model_validate_json(payload) if payload else None

    async def save(self, conversation: ConversationState) -> None:
        payload = conversation.model_dump_json()

        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO conversations (conversation_id, payload) VALUES (?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET payload = excluded.payload
                """,
                (conversation.conversation_id, payload),
            )
            connection.commit()

        await self._database.call(write)


class SqliteRunStore:
    def __init__(self, database: _Database) -> None:
        self._database = database

    async def append(self, run: RunRecord) -> None:
        values = (
            run.run_id,
            run.conversation_id,
            run.agent_id,
            run.sop_id,
            run.kind.value,
            run.event,
            run.status.value,
            run.outcome.value,
            run.reason,
            run.created_at.isoformat(),
        )

        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, conversation_id, agent_id, sop_id, kind,
                    event, status, outcome, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            connection.commit()

        await self._database.call(write)

    async def get(self, run_id: str) -> RunRecord | None:
        def read(connection: sqlite3.Connection) -> sqlite3.Row | None:
            return connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

        row = await self._database.call(read)
        return _run_from(row) if row else None

    async def list_for(self, conversation_id: str) -> list[RunRecord]:
        def read(connection: sqlite3.Connection) -> list[sqlite3.Row]:
            return connection.execute(
                """
                SELECT * FROM runs
                WHERE conversation_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [_run_from(row) for row in await self._database.call(read)]


def _run_from(row: sqlite3.Row) -> RunRecord:
    return RunRecord.model_validate(dict(row))


def open_sqlite(path: str) -> tuple[ConversationStore, RunStore]:
    """Open one SQLite connection shared by the two persistence adapters."""
    database = _Database(path)
    return SqliteConversationStore(database), SqliteRunStore(database)
