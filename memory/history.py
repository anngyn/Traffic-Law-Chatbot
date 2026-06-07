# memory/history.py — persistent chat history (SQLAlchemy; sqlite default, postgres via env).
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

_DEFAULT_DB = (Path(__file__).resolve().parents[1] / "data" / "chat_history.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def save_message(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as s:
        s.add(ChatMessage(session_id=session_id, role=role, content=content))
        s.commit()


def get_history(session_id: str, limit: int = 10) -> list[tuple[str, str]]:
    with SessionLocal() as s:
        rows = (
            s.query(ChatMessage)
            .filter_by(session_id=session_id)
            .order_by(ChatMessage.id.desc())
            .limit(limit)
            .all()
        )
    return [(r.role, r.content) for r in reversed(rows)]
