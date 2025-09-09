#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Doggy Diary — Streamlit app for dog hotels (multi-tenant)
# Profiles, bookings (recurring, discounts, paid), calendar (grid + timelines),
# pricing, capacity/daily limits, insights, export to .ics.
# Adds: Account recovery + Contact Us + Admin dashboard, Streamlit API updates.

from __future__ import annotations

import os, smtplib, random
import calendar as pycal
import math
from email.message import EmailMessage
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from dateutil import tz
from icalendar import Calendar, Event, Alarm
from passlib.hash import bcrypt
from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, Float, Integer, Boolean,
    ForeignKey, UniqueConstraint, select, func, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

# ---------------- App config / constants ----------------
st.set_page_config(page_title="Doggy Diary", page_icon="🐶", layout="wide")

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTBOX_DIR = DATA_DIR / "outbox"
for p in (DATA_DIR, UPLOAD_DIR, OUTBOX_DIR):
    p.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{(DATA_DIR / 'doggy_diary.db').as_posix()}"

DEFAULT_TZ = "Europe/London"
SERVICE_TYPES = ["walk", "daycare", "overnight", "home_visit"]
DEFAULT_CAPACITY = {"walk": 4, "daycare": 6, "overnight": 2, "home_visit": 3}

# Support & app URL (kept hidden in UI)
SUPPORT_TO = os.environ.get("SUPPORT_TO", "evancawley@outlook.com")
try:
    # Avoid hard fail if secrets.toml is not present
    APP_BASE_URL = st.secrets.get("app_base_url")
except Exception:
    APP_BASE_URL = os.environ.get("APP_BASE_URL")

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# ---------------- Models ----------------
class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AppSetting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    owner_id = Column(String, index=True)
    tz_name = Column(String, default=DEFAULT_TZ)
    alarm_minutes = Column(Integer, default=15)
    sibling_discount_percent = Column(Integer, default=20)
    dur_walk_min = Column(Integer, default=60)
    dur_daycare_min = Column(Integer, default=8 * 60)
    dur_overnight_min = Column(Integer, default=24 * 60)
    dur_home_visit_min = Column(Integer, default=60)


class Capacity(Base):
    __tablename__ = "capacities"
    __table_args__ = (UniqueConstraint("owner_id", "service_type", name="uq_capacity_owner_service"),)
    id = Column(Integer, primary_key=True)
    owner_id = Column(String, index=True)
    service_type = Column(String, nullable=False)
    max_dogs = Column(Integer, nullable=False)


class DailyCap(Base):
    __tablename__ = "daily_caps"
    __table_args__ = (UniqueConstraint("owner_id", "service_type", name="uq_dailycap_owner_service"),)
    id = Column(Integer, primary_key=True)
    owner_id = Column(String, index=True)
    service_type = Column(String, nullable=False)
    max_per_day = Column(Integer)  # 0/NULL => unlimited


class OwnerProfile(Base):
    __tablename__ = "owner_profile"
    id = Column(Integer, primary_key=True)
    owner_id = Column(String, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)


class Dog(Base):
    __tablename__ = "dogs"
    id = Column(String, primary_key=True)
    owner_id = Column(String, index=True)
    name = Column(String, nullable=False, index=True)
    breed = Column(String)
    sex = Column(String)
    dob = Column(DateTime(timezone=True))
    weight_kg = Column(Float)
    vet_name = Column(String)
    vet_phone = Column(String)
    meds_notes = Column(Text)
    diet_notes = Column(Text)
    general_notes = Column(Text)
    photo_path = Column(String)
    household = Column(String)
    price_walk = Column(Float)
    price_daycare = Column(Float)
    price_overnight = Column(Float)
    price_home_visit = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    bookings = relationship("Booking", back_populates="dog", cascade="all, delete-orphan")


class Booking(Base):
    __tablename__ = "bookings"
    id = Column(String, primary_key=True)
    owner_id = Column(String, index=True)
    dog_id = Column(String, ForeignKey("dogs.id", ondelete="CASCADE"), index=True, nullable=False)
    service_type = Column(String, nullable=False)  # walk/daycare/overnight/home_visit
    status = Column(String, default="booked")      # booked/pending/cancelled
    start_utc = Column(DateTime(timezone=True), nullable=False)
    end_utc = Column(DateTime(timezone=True), nullable=False)
    location = Column(String)

    price_before_discount = Column(Float)
    discount_type = Column(String)  # none|percent|amount|override
    discount_value = Column(Float)
    discount_amount = Column(Float)
    sibling_discount_applied = Column(Boolean, default=False)
    price = Column(Float)

    paid = Column(Boolean, default=False)
    paid_at = Column(DateTime(timezone=True))
    notes = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    dog = relationship("Dog", back_populates="bookings")


class PasswordReset(Base):
    __tablename__ = "password_resets"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    token = Column(String, unique=True, index=True)
    code = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    used_at = Column(DateTime(timezone=True))


Base.metadata.create_all(bind=engine)


# ---------- Migrations (robust) ----------
from sqlalchemy import text as _sql_text

def _table_sql(conn, name: str) -> str | None:
    row = conn.execute(_sql_text(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=:n"
    ), {"n": name}).fetchone()
    return row[0] if row else None

def _table_exists(conn, name: str) -> bool:
    return conn.execute(_sql_text(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"
    ), {"n": name}).fetchone() is not None

def _has_col(conn, table: str, col: str) -> bool:
    return any(r[1] == col for r in conn.execute(_sql_text(f"PRAGMA table_info({table})")).fetchall())

def _unique_indexes(conn, table: str):
    out = []
    for row in conn.execute(_sql_text(f"PRAGMA index_list({table})")).fetchall():
        idx_name = row[1]
        is_unique = bool(row[2])
        cols = [r[2] for r in conn.execute(_sql_text(f"PRAGMA index_info({idx_name})")).fetchall()]
        out.append((idx_name, is_unique, cols))
    return out

def _needs_rebuild_capacities(conn) -> bool:
    if not _table_exists(conn, "capacities"): return False
    if not _has_col(conn, "capacities", "owner_id"): return True
    sql = _table_sql(conn, "capacities") or ""
    if "UNIQUE" in sql and "owner_id" not in sql and "service_type" in sql: return True
    for _, is_unique, cols in _unique_indexes(conn, "capacities"):
        if is_unique and cols == ["service_type"]: return True
    return False

def _needs_rebuild_daily_caps(conn) -> bool:
    if not _table_exists(conn, "daily_caps"): return True
    if not _has_col(conn, "daily_caps", "owner_id"): return True
    sql = _table_sql(conn, "daily_caps") or ""
    if "UNIQUE" in sql and "owner_id" not in sql and "service_type" in sql: return True
    for _, is_unique, cols in _unique_indexes(conn, "daily_caps"):
        if is_unique and cols == ["service_type"]: return True
    return False

def _rebuild_capacities(conn):
    conn.execute(_sql_text("PRAGMA foreign_keys=off"))
    conn.execute(_sql_text("""
        CREATE TABLE capacities_new (
            id INTEGER PRIMARY KEY,
            owner_id TEXT,
            service_type TEXT NOT NULL,
            max_dogs INTEGER NOT NULL,
            UNIQUE(owner_id, service_type)
        )
    """))
    if _table_exists(conn, "capacities"):
        conn.execute(_sql_text("""
            INSERT OR IGNORE INTO capacities_new (id, owner_id, service_type, max_dogs)
            SELECT id, owner_id, service_type, max_dogs FROM capacities
        """))
        conn.execute(_sql_text("DROP TABLE capacities"))
    conn.execute(_sql_text("ALTER TABLE capacities_new RENAME TO capacities"))
    conn.execute(_sql_text("PRAGMA foreign_keys=on"))

def _rebuild_daily_caps(conn):
    conn.execute(_sql_text("PRAGMA foreign_keys=off"))
    conn.execute(_sql_text("""
        CREATE TABLE daily_caps_new (
            id INTEGER PRIMARY KEY,
            owner_id TEXT,
            service_type TEXT NOT NULL,
            max_per_day INTEGER,
            UNIQUE(owner_id, service_type)
        )
    """))
    if _table_exists(conn, "daily_caps"):
        conn.execute(_sql_text("""
            INSERT OR IGNORE INTO daily_caps_new (id, owner_id, service_type, max_per_day)
            SELECT id, owner_id, service_type, max_per_day FROM daily_caps
        """))
        conn.execute(_sql_text("DROP TABLE daily_caps"))
    conn.execute(_sql_text("ALTER TABLE daily_caps_new RENAME TO daily_caps"))
    conn.execute(_sql_text("PRAGMA foreign_keys=on"))

def run_migrations():
    with engine.begin() as conn:
        # Ensure required columns
        for t in ["settings", "capacities", "daily_caps", "owner_profile", "dogs", "bookings"]:
            if _table_exists(conn, t) and not _has_col(conn, t, "owner_id"):
                conn.execute(_sql_text(f"ALTER TABLE {t} ADD COLUMN owner_id TEXT"))

        if _table_exists(conn, "settings"):
            for c, typ in [
                ("sibling_discount_percent", "INTEGER"),
                ("dur_walk_min", "INTEGER"),
                ("dur_daycare_min", "INTEGER"),
                ("dur_overnight_min", "INTEGER"),
                ("dur_home_visit_min", "INTEGER"),
                ("alarm_minutes", "INTEGER"),
                ("tz_name", "TEXT"),
            ]:
                if not _has_col(conn, "settings", c):
                    conn.execute(_sql_text(f"ALTER TABLE settings ADD COLUMN {c} {typ}"))

        if _table_exists(conn, "dogs"):
            for c, typ in [
                ("household", "TEXT"),
                ("price_walk", "REAL"),
                ("price_daycare", "REAL"),
                ("price_overnight", "REAL"),
                ("price_home_visit", "REAL"),
            ]:
                if not _has_col(conn, "dogs", c):
                    conn.execute(_sql_text(f"ALTER TABLE dogs ADD COLUMN {c} {typ}"))

        if _table_exists(conn, "bookings"):
            for c, typ in [
                ("price_before_discount", "REAL"),
                ("discount_type", "TEXT"),
                ("discount_value", "REAL"),
                ("discount_amount", "REAL"),
                ("sibling_discount_applied", "INTEGER"),
                ("paid", "INTEGER"),
                ("paid_at", "TIMESTAMP"),
            ]:
                if not _has_col(conn, "bookings", c):
                    conn.execute(_sql_text(f"ALTER TABLE bookings ADD COLUMN {c} {typ}"))
            conn.execute(_sql_text("UPDATE bookings SET status='pending' WHERE status='tentative'"))

        # Build capacities/daily_caps with composite uniques
        if _needs_rebuild_capacities(conn): _rebuild_capacities(conn)
        if _needs_rebuild_daily_caps(conn): _rebuild_daily_caps(conn)

        # De-dupe singleton tables and enforce one-per-owner
        conn.execute(_sql_text("""
            DELETE FROM settings
            WHERE owner_id IS NOT NULL
              AND id NOT IN (
                SELECT MAX(id) FROM settings WHERE owner_id IS NOT NULL GROUP BY owner_id
              )
        """))
        conn.execute(_sql_text("""
            DELETE FROM owner_profile
            WHERE owner_id IS NOT NULL
              AND id NOT IN (
                SELECT MAX(id) FROM owner_profile WHERE owner_id IS NOT NULL GROUP BY owner_id
              )
        """))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_settings_owner ON settings(owner_id)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_owner_profile_owner ON owner_profile(owner_id)"))

        # Helpful indexes
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_owner ON bookings(owner_id)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_time ON bookings(start_utc, end_utc)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_dogs_owner ON dogs(owner_id)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_capacity_owner_service ON capacities(owner_id, service_type)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_dailycap_owner_service ON daily_caps(owner_id, service_type)"))

        # Password resets table (if missing)
        if not _table_exists(conn, "password_resets"):
            conn.execute(_sql_text("""
                CREATE TABLE password_resets (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT UNIQUE,
                    code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    used_at TIMESTAMP
                )
            """))
            conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_pwreset_user ON password_resets(user_id)"))
            conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_pwreset_token ON password_resets(token)"))

run_migrations()


# ---------------- Email helpers ----------------
def _smtp_creds_ok():
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"))

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Return True if sent over SMTP; otherwise save to outbox file and return False."""
    try:
        if not _smtp_creds_ok():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            f = OUTBOX_DIR / f"mail-{ts}.txt"
            f.write_text(f"TO: {to_email}\nSUBJECT: {subject}\n\n{body}", encoding="utf-8")
            return False
        host = os.environ.get("SMTP_HOST"); port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER"); pwd = os.environ.get("SMTP_PASS")
        use_tls = os.environ.get("SMTP_TLS", "1") not in ("0", "false", "False")
        from_addr = os.environ.get("SMTP_FROM", user)
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=20) as s:
            if use_tls: s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        f = OUTBOX_DIR / f"mail-{ts}.txt"
        f.write_text(f"TO: {to_email}\nSUBJECT: {subject}\n\n{body}", encoding="utf-8")
        return False


# ---------------- Auth & defaults ----------------
def new_id() -> str:
    import uuid
    return uuid.uuid4().hex

def hash_pwd(p: str) -> str:
    return bcrypt.hash(p)

def verify_pwd(p: str, h: str) -> bool:
    try:
        return bcrypt.verify(p, h)
    except Exception:
        return False

def current_user_id() -> Optional[str]:
    return st.session_state.get("user_id")

def is_admin() -> bool:
    return bool(st.session_state.get("is_admin"))

def ensure_user_defaults(owner_id: str, email: str, *, db: Optional[Session] = None):
    """Create or adopt per-owner defaults, and hard-dedupe duplicates so later lookups are safe."""
    close = False
    if db is None:
        db = SessionLocal()
        close = True
    try:
        # SETTINGS singleton
        rows = db.execute(
            select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())
        ).scalars().all()
        if rows:
            for extra in rows[1:]:
                db.delete(extra)
            if not rows[0].tz_name:
                rows[0].tz_name = DEFAULT_TZ
        else:
            legacy = db.execute(
                select(AppSetting).where(AppSetting.owner_id.is_(None)).order_by(AppSetting.id.desc())
            ).scalars().first()
            if legacy:
                legacy.owner_id = owner_id
                leftovers = db.execute(
                    select(AppSetting).where(AppSetting.owner_id.is_(None), AppSetting.id != legacy.id)
                ).scalars().all()
                for x in leftovers:
                    db.delete(x)
            else:
                db.add(AppSetting(
                    owner_id=owner_id, tz_name=DEFAULT_TZ, alarm_minutes=15,
                    sibling_discount_percent=20, dur_walk_min=60, dur_daycare_min=480,
                    dur_overnight_min=1440, dur_home_visit_min=60
                ))

        # OWNER PROFILE singleton
        rows_p = db.execute(
            select(OwnerProfile).where(OwnerProfile.owner_id == owner_id).order_by(OwnerProfile.id.desc())
        ).scalars().all()
        if rows_p:
            for extra in rows_p[1:]:
                db.delete(extra)
        else:
            legacy_p = db.execute(
                select(OwnerProfile).where(OwnerProfile.owner_id.is_(None)).order_by(OwnerProfile.id.desc())
            ).scalars().first()
            if legacy_p:
                legacy_p.owner_id = owner_id
                leftovers_p = db.execute(
                    select(OwnerProfile).where(OwnerProfile.owner_id.is_(None), OwnerProfile.id != legacy_p.id)
                ).scalars().all()
                for x in leftovers_p:
                    db.delete(x)
            else:
                db.add(OwnerProfile(owner_id=owner_id, name="", email=email, phone=""))

        # Capacities & daily caps
        existing = {c.service_type: c for c in db.execute(
            select(Capacity).where(Capacity.owner_id == owner_id)
        ).scalars()}
        for stype, cap in DEFAULT_CAPACITY.items():
            if stype not in existing:
                db.add(Capacity(owner_id=owner_id, service_type=stype, max_dogs=cap))

        existing_d = {c.service_type: c for c in db.execute(
            select(DailyCap).where(DailyCap.owner_id == owner_id)
        ).scalars()}
        for stype in SERVICE_TYPES:
            if stype not in existing_d:
                db.add(DailyCap(owner_id=owner_id, service_type=stype, max_per_day=None))

        db.commit()
    finally:
        if close:
            db.close()

def adopt_legacy_rows(owner_id: str, *, db: Optional[Session] = None):
    """Adopt legacy rows without violating composite uniques; dedupe singletons."""
    close = False
    if db is None:
        db = SessionLocal()
        close = True
    try:
        # capacities
        db.execute(text("""
            UPDATE capacities
               SET owner_id = :oid
             WHERE owner_id IS NULL
               AND service_type NOT IN (SELECT service_type FROM capacities WHERE owner_id = :oid)
        """), {"oid": owner_id})
        db.execute(text("DELETE FROM capacities WHERE owner_id IS NULL"))

        # daily_caps
        db.execute(text("""
            UPDATE daily_caps
               SET owner_id = :oid
             WHERE owner_id IS NULL
               AND service_type NOT IN (SELECT service_type FROM daily_caps WHERE owner_id = :oid)
        """), {"oid": owner_id})
        db.execute(text("DELETE FROM daily_caps WHERE owner_id IS NULL"))

        # settings singleton
        existing = db.execute(
            select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())
        ).scalars().all()
        nulls = db.execute(
            select(AppSetting).where(AppSetting.owner_id.is_(None)).order_by(AppSetting.id.desc())
        ).scalars().all()
        if existing:
            for r in nulls:
                db.delete(r)
        else:
            if nulls:
                keep = nulls[0]
                keep.owner_id = owner_id
                for r in nulls[1:]:
                    db.delete(r)

        # owner profile singleton
        existing_p = db.execute(
            select(OwnerProfile).where(OwnerProfile.owner_id == owner_id).order_by(OwnerProfile.id.desc())
        ).scalars().all()
        nulls_p = db.execute(
            select(OwnerProfile).where(OwnerProfile.owner_id.is_(None)).order_by(OwnerProfile.id.desc())
        ).scalars().all()
        if existing_p:
            for r in nulls_p:
                db.delete(r)
        else:
            if nulls_p:
                keep = nulls_p[0]
                keep.owner_id = owner_id
                for r in nulls_p[1:]:
                    db.delete(r)

        # mass adopt others
        for table in ["dogs", "bookings"]:
            db.execute(text(f"UPDATE {table} SET owner_id=:oid WHERE owner_id IS NULL"), {"oid": owner_id})

        db.commit()
    finally:
        if close:
            db.close()


# ---------------- Helper funcs ----------------
def get_settings(db: Session, owner_id: str) -> AppSetting:
    rows = db.execute(
        select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())
    ).scalars().all()
    if not rows:
        s = AppSetting(
            owner_id=owner_id, tz_name=DEFAULT_TZ, alarm_minutes=15,
            sibling_discount_percent=20, dur_walk_min=60, dur_daycare_min=480,
            dur_overnight_min=1440, dur_home_visit_min=60
        )
        db.add(s)
        db.commit()
        return s
    primary = rows[0]
    for extra in rows[1:]:
        db.delete(extra)
    db.commit()
    return primary

def _aware_utc(dt):
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def to_utc(dt_local: datetime, tz_name: str) -> datetime:
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=tz.gettz(tz_name))
    return dt_local.astimezone(timezone.utc)

def to_local(dt_utc: datetime, tz_name: str) -> datetime:
    return dt_utc.astimezone(tz.gettz(tz_name) or tz.gettz(DEFAULT_TZ))

def sweepline_max_concurrency(intervals: List[Tuple[datetime, datetime]]) -> int:
    pts = []
    for s, e in intervals:
        pts.append((_aware_utc(s), +1))
        pts.append((_aware_utc(e), -1))
    pts.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    mx = 0
    for _, d in pts:
        cur += d
        mx = max(mx, cur)
    return mx

def check_capacity(db: Session, owner_id: str, service_type: str,
                   start_utc: datetime, end_utc: datetime,
                   exclude_booking_id: Optional[str] = None):
    cap_row = db.execute(
        select(Capacity).where(Capacity.owner_id == owner_id, Capacity.service_type == service_type)
    ).scalar_one_or_none()
    if not cap_row:
        cap_row = Capacity(owner_id=owner_id, service_type=service_type,
                           max_dogs=DEFAULT_CAPACITY.get(service_type, 5))
        db.add(cap_row)
        db.commit()
    cap = cap_row.max_dogs
    q = select(Booking).where(
        Booking.owner_id == owner_id,
        Booking.service_type == service_type,
        Booking.status == "booked",
        Booking.start_utc < end_utc,
        Booking.end_utc > start_utc,
    )
    if exclude_booking_id:
        q = q.where(Booking.id != exclude_booking_id)
    overlaps = list(db.execute(q).scalars())
    peak = sweepline_max_concurrency([(b.start_utc, b.end_utc) for b in overlaps] + [(start_utc, end_utc)])
    return peak <= cap, peak, cap, overlaps

def days_covered_local(s_utc: datetime, e_utc: datetime, tz_name: str) -> List[date]:
    s = to_local(s_utc, tz_name)
    e = to_local(e_utc, tz_name)
    d = s.date()
    last = e.date()
    out = []
    while d <= last:
        out.append(d)
        d += timedelta(days=1)
    return out

def check_daily_limit(db: Session, owner_id: str, service: str,
                      s_utc: datetime, e_utc: datetime, tz_name: str,
                      exclude_id: Optional[str] = None):
    row = db.execute(select(DailyCap).where(DailyCap.owner_id == owner_id, DailyCap.service_type == service)).scalar_one_or_none()
    limit = row.max_per_day if row else None
    if not limit or limit <= 0:
        return True, {}
    exceeded = {}
    for d in days_covered_local(s_utc, e_utc, tz_name):
        zone = tz.gettz(tz_name)
        s_l = datetime.combine(d, time.min, zone)
        e_l = datetime.combine(d, time.max, zone)
        sU = s_l.astimezone(timezone.utc)
        eU = e_l.astimezone(timezone.utc)
        q = select(func.count(Booking.id)).where(
            Booking.owner_id == owner_id, Booking.service_type == service,
            Booking.status == "booked", Booking.start_utc < eU, Booking.end_utc > sU
        )
        if exclude_id:
            q = q.where(Booking.id != exclude_id)
        n = db.execute(q).scalar_one() or 0
        if n + 1 > limit:
            exceeded[d] = (n + 1, limit)
    return (len(exceeded) == 0), exceeded

def price_for_booking(dog: 'Dog', service: str, s_local: datetime, e_local: datetime) -> float:
    if service == "overnight":
        base = dog.price_overnight or 0.0
        seconds = max((e_local - s_local).total_seconds(), 0.0)
        blocks = max(1, math.ceil(seconds / (24 * 3600)))
        return round(base * blocks, 2)
    return round({
        "walk": dog.price_walk or 0.0,
        "daycare": dog.price_daycare or 0.0,
        "home_visit": dog.price_home_visit or 0.0
    }.get(service, 0.0), 2)

def overlapping_sibling_count(db: Session, owner_id: str, dog: 'Dog', s_utc: datetime, e_utc: datetime) -> int:
    if not dog.household:
        return 0
    q = (select(Booking, Dog).join(Dog, Dog.id == Booking.dog_id)
         .where(Dog.owner_id == owner_id, Booking.owner_id == owner_id, Dog.household == dog.household,
                Booking.dog_id != dog.id, Booking.status == "booked",
                Booking.start_utc < e_utc, Booking.end_utc > s_utc))
    return len(db.execute(q).all())

def bookings_df(db: Session, owner_id: str, tz_name: str,
                start: datetime | None = None, end: datetime | None = None,
                dog_id: str | None = None, service: str | None = None,
                statuses: List[str] | None = None, paid_filter: str | None = None) -> pd.DataFrame:
    q = select(Booking, Dog).join(Dog, Dog.id == Booking.dog_id).where(Booking.owner_id == owner_id, Dog.owner_id == owner_id)
    if statuses:
        q = q.where(Booking.status.in_(tuple(statuses)))
    if start and end:
        q = q.where(Booking.start_utc < end, Booking.end_utc > start)
    if dog_id:
        q = q.where(Booking.dog_id == dog_id)
    if service:
        q = q.where(Booking.service_type == service)
    if paid_filter == "paid":
        q = q.where(Booking.paid == True)
    if paid_filter == "unpaid":
        q = q.where((Booking.paid == False) | (Booking.paid.is_(None)))
    rows = [(b, d) for b, d in db.execute(q).all()]
    data = [{
        "ID": b.id, "Dog": d.name, "DogID": d.id, "Household": d.household, "Service": b.service_type,
        "Status": b.status, "Paid": bool(b.paid),
        "Start (local)": to_local(b.start_utc, tz_name), "End (local)": to_local(b.end_utc, tz_name),
        "Price (£)": b.price, "Discount (£)": b.discount_amount,
        "Location": b.location, "Notes": b.notes
    } for b, d in rows]
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by=["Start (local)", "Dog"]).reset_index(drop=True)
    return df

def _timeline_plot(df: pd.DataFrame, group_by: str) -> None:
    plot_df = pd.DataFrame({
        "Start": df["Start (local)"], "Finish": df["End (local)"],
        "Service": df["Service"].str.title(), "Dog": df["Dog"],
        "Resource": df["Dog"] if group_by == "Dog" else df["Service"].str.title(),
        "Details": df["Location"].fillna(""),
    })
    fig = px.timeline(plot_df, x_start="Start", x_end="Finish", y="Resource",
                      color="Service", hover_data=["Dog", "Service", "Start", "Finish", "Details"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=560, margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------- UI: topbar & nav ----------------
def topbar():
    cols = st.columns([0.07, 0.73, 0.20])
    with cols[0]:
        if st.button("👤", key="topbar_profile_btn", help="My Profile"):
            st.session_state.page = "My Profile"
            st.rerun()
    with cols[2]:
        if is_admin() and st.session_state.get("impersonated_by_admin"):
            if st.button("Exit impersonation", key="exit_imp_btn"):
                # keep admin, drop impersonation
                st.session_state.pop("user_id", None)
                st.session_state.pop("user_name", None)
                st.session_state.pop("impersonated_by_admin", None)
                st.session_state.page = "Admin"
                st.rerun()
        if st.button("Log out", key="logout_btn"):
            # full logout
            keys = list(st.session_state.keys())
            for k in keys:
                st.session_state.pop(k, None)
            st.rerun()

def nav_home():
    topbar()
    st.title("🐶 Doggy Diary")
    st.caption("Manage dog profiles, bookings, and your calendar at a glance.")
    c1, c2, c3 = st.columns(3)
    if c1.button("🦴  Doggy Profiles", key="home_profiles_btn", use_container_width=True, type="primary"):
        st.session_state.page = "Doggy Profiles"; st.rerun()
    if c2.button("📅  Bookings", key="home_bookings_btn", use_container_width=True, type="primary"):
        st.session_state.page = "Bookings"; st.rerun()
    if c3.button("🗓️  Calendar", key="home_calendar_btn", use_container_width=True, type="primary"):
        st.session_state.page = "Calendar"; st.rerun()
    st.markdown("---")
    st.write("Tip: Set **daily limits** and **concurrent capacity** in **Settings**.")


# ---------------- Doggy Profiles ----------------
def dogs_section():
    if not current_user_id():
        auth_page(); return
    topbar()
    st.header("🐾 Doggy Profiles")
    tab_add, tab_manage = st.tabs(["Add New", "Manage Existing"])
    owner_id = current_user_id()
    with SessionLocal() as db:
        with tab_add:
            with st.form("add_dog_form", clear_on_submit=True):
                cols = st.columns(3)
                name = cols[0].text_input("Name *")
                breed = cols[1].text_input("Breed")
                sex = cols[2].selectbox("Sex", ["Unknown", "Male", "Female"], index=0)
                c2 = st.columns(3)
                dob = c2[0].date_input("Date of Birth", format="DD/MM/YYYY")
                weight = c2[1].number_input("Weight (kg)", min_value=0.0, step=0.1, value=0.0)
                photo = c2[2].file_uploader("Photo", type=["png", "jpg", "jpeg", "webp"])
                hh = st.text_input("Household (for siblings)")
                st.subheader("Pricing (per dog)")
                p = st.columns(4)
                price_walk = p[0].number_input("Walk (£/booking)", min_value=0.0, step=0.5, value=0.0)
                price_day = p[1].number_input("Daycare (£/booking)", min_value=0.0, step=0.5, value=0.0)
                price_over = p[2].number_input("Overnight (£/24h block)", min_value=0.0, step=0.5, value=0.0)
                price_home = p[3].number_input("Home Visit (£/booking)", min_value=0.0, step=0.5, value=0.0)
                vet_name = st.text_input("Vet name")
                vet_phone = st.text_input("Vet phone")
                meds = st.text_area("Medication notes")
                diet = st.text_area("Dietary notes")
                notes = st.text_area("General notes")
                if st.form_submit_button("Add dog", use_container_width=True):
                    if not name.strip():
                        st.error("Dog name is required.")
                    else:
                        tz_name = get_settings(db, owner_id).tz_name
                        d = Dog(
                            id=new_id(), owner_id=owner_id, name=name.strip(), breed=breed.strip() or None, sex=sex,
                            dob=datetime.combine(dob, time.min, tz.gettz(tz_name)) if dob else None,
                            weight_kg=float(weight) if weight else None, vet_name=vet_name.strip() or None,
                            vet_phone=vet_phone.strip() or None, meds_notes=meds.strip() or None,
                            diet_notes=diet.strip() or None, general_notes=notes.strip() or None,
                            household=hh.strip() or None, price_walk=float(price_walk),
                            price_daycare=float(price_day), price_overnight=float(price_over), price_home_visit=float(price_home)
                        )
                        if photo is not None:
                            ext = Path(photo.name).suffix.lower()
                            save = UPLOAD_DIR / f"{d.id}{ext}"
                            Image.open(photo).save(save)
                            d.photo_path = str(save)
                        db.add(d); db.commit(); st.success(f"Added {d.name} ✅")
        with tab_manage:
            dogs = list(db.execute(select(Dog).where(Dog.owner_id == owner_id).order_by(Dog.name)).scalars())
            if not dogs:
                st.info("No dogs yet. Add one in the **Add New** tab."); return
            idx = st.selectbox("Select a dog", options=list(range(len(dogs))),
                               format_func=lambda i: f"{dogs[i].name} ({dogs[i].breed or 'Unknown'})")
            d = dogs[idx]
            cols = st.columns(2)
            with cols[0]:
                if d.photo_path and Path(d.photo_path).exists():
                    st.image(d.photo_path, caption=d.name, use_column_width=True)
                else:
                    st.info("No photo uploaded.")
            with cols[1]:
                with st.form(f"edit_dog_{d.id}"):
                    e_name = st.text_input("Name *", value=d.name)
                    e_breed = st.text_input("Breed", value=d.breed or "")
                    e_sex = st.selectbox("Sex", ["Unknown", "Male", "Female"],
                                         index=["Unknown", "Male", "Female"].index(d.sex or "Unknown"))
                    e_dob = st.date_input("Date of Birth", value=d.dob.date() if d.dob else None, format="DD/MM/YYYY")
                    e_weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1, value=d.weight_kg or 0.0)
                    e_hh = st.text_input("Household (for siblings)", value=d.household or "")
                    st.subheader("Pricing")
                    p = st.columns(4)
                    e_pw = p[0].number_input("Walk (£/booking)", min_value=0.0, step=0.5, value=d.price_walk or 0.0)
                    e_pd = p[1].number_input("Daycare (£/booking)", min_value=0.0, step=0.5, value=d.price_daycare or 0.0)
                    e_po = p[2].number_input("Overnight (£/24h)", min_value=0.0, step=0.5, value=d.price_overnight or 0.0)
                    e_ph = p[3].number_input("Home Visit (£/booking)", min_value=0.0, step=0.5, value=d.price_home_visit or 0.0)
                    e_vn = st.text_input("Vet name", value=d.vet_name or "")
                    e_vp = st.text_input("Vet phone", value=d.vet_phone or "")
                    e_m = st.text_area("Medication notes", value=d.meds_notes or "")
                    e_di = st.text_area("Dietary notes", value=d.diet_notes or "")
                    e_no = st.text_area("General notes", value=d.general_notes or "")
                    e_photo = st.file_uploader("Replace photo", type=["png", "jpg", "jpeg", "webp"], key=f"photo_{d.id}")
                    c1, c2 = st.columns(2)
                    save = c1.form_submit_button("Save changes", use_container_width=True)
                    delete = c2.form_submit_button("Delete dog", use_container_width=True)
                if save:
                    tz_name = get_settings(db, owner_id).tz_name
                    d.name = e_name.strip() or d.name
                    d.breed = e_breed.strip() or None
                    d.sex = e_sex
                    if e_dob:
                        d.dob = datetime.combine(e_dob, time.min, tz.gettz(tz_name))
                    d.weight_kg = float(e_weight) if e_weight else None
                    d.vet_name = e_vn.strip() or None
                    d.vet_phone = e_vp.strip() or None
                    d.meds_notes = e_m.strip() or None
                    d.diet_notes = e_di.strip() or None
                    d.general_notes = e_no.strip() or None
                    d.household = e_hh.strip() or None
                    d.price_walk = float(e_pw)
                    d.price_daycare = float(e_pd)
                    d.price_overnight = float(e_po)
                    d.price_home_visit = float(e_ph)
                    if e_photo is not None:
                        ext = Path(e_photo.name).suffix.lower()
                        save = UPLOAD_DIR / f"{d.id}{ext}"
                        Image.open(e_photo).save(save)
                        d.photo_path = str(save)
                    db.commit(); st.success("Saved ✅")
                if delete:
                    db.delete(d); db.commit()
                    st.warning("Dog deleted."); st.rerun()


# ---------------- My Profile & Insights ----------------
def my_profile_section():
    if not current_user_id():
        auth_page(); return
    topbar()
    st.header("👤 My Profile & Earnings")
    owner_id = current_user_id()
    with SessionLocal() as db:
        owner = db.execute(select(OwnerProfile).where(OwnerProfile.owner_id == owner_id)).scalar_one_or_none()
        if not owner:
            owner = OwnerProfile(owner_id=owner_id, name="", email="", phone="")
            db.add(owner); db.commit()
        with st.form("owner_form"):
            c = st.columns(3)
            name = c[0].text_input("Your name", value=owner.name or "")
            email = c[1].text_input("Email", value=owner.email or "")
            phone = c[2].text_input("Phone", value=owner.phone or "")
            if st.form_submit_button("Save profile", use_container_width=True):
                owner.name, owner.email, owner.phone = name.strip(), email.strip(), phone.strip()
                db.commit(); st.success("Profile saved ✅")

        st.subheader("Earnings & Activity")
        tz_name = get_settings(db, owner_id).tz_name

        # Entire current year by default (1 Jan → 31 Dec)
        this_year = datetime.now().year
        default_start = date(this_year, 1, 1)
        default_end = date(this_year, 12, 31)  # full year
        custom = st.checkbox("Use custom dates", value=False, key="prof_custom_dates")
        if custom:
            c = st.columns(3)
            start_day = c[0].date_input("From", value=default_start)
            end_day = c[1].date_input("To", value=default_end)
            paid_only = c[2].checkbox("Paid only", value=False)
        else:
            start_day, end_day, paid_only = default_start, default_end, False
            st.caption(f"Showing **{default_start} → {default_end}** (entire current year)")

        sU = to_utc(datetime.combine(start_day, time.min), tz_name)
        eU = to_utc(datetime.combine(end_day, time.max), tz_name)

        statuses = ["booked", "pending"]
        df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, statuses=statuses,
                         paid_filter=("paid" if paid_only else None))

        paid_sum = df.loc[df["Paid"] == True, "Price (£)"].sum() if not df.empty else 0.0
        unpaid_booked = df.loc[(df["Paid"] == False) & (df["Status"] == "booked"), "Price (£)"].sum() if not df.empty else 0.0
        pending_sum = df.loc[df["Status"] == "pending", "Price (£)"].sum() if not df.empty else 0.0
        total_selected = 0.0 if df.empty else df["Price (£)"].sum()

        m = st.columns(4)
        m[0].metric("Paid (selected)", f"£{paid_sum:,.2f}")
        m[1].metric("Unpaid (Booked)", f"£{unpaid_booked:,.2f}")
        m[2].metric("Pending (Quotes)", f"£{pending_sum:,.2f}")
        m[3].metric("Expected (selected)", f"£{total_selected:,.2f}")

        if not df.empty:
            st.subheader("Most Booked Dogs")
            st.dataframe(
                df.groupby("Dog")["ID"].count().sort_values(ascending=False)
                .reset_index().rename(columns={"ID": "Bookings"}),
                use_container_width=True
            )
            st.subheader("Popular Booking Types")
            st.dataframe(
                df.groupby("Service")["ID"].count().sort_values(ascending=False)
                .reset_index().rename(columns={"ID": "Bookings"}),
                use_container_width=True
            )

            st.subheader("Outstanding (Unpaid, Booked)")
            outstanding = df[(df["Status"] == "booked") & (df["Paid"] == False)]
            if outstanding.empty:
                st.info("No outstanding items 🎉")
            else:
                for _, r in outstanding.iterrows():
                    with st.container():
                        cols = st.columns([0.3, 0.28, 0.22, 0.1, 0.1])
                        cols[0].write(f"**{r['Dog']}** — {r['Service']}")
                        cols[1].write(f"{r['Start (local)']:%d %b %H:%M} → {r['End (local)']:%d %b %H:%M}")
                        cols[2].write(f"£{(r['Price (£)'] or 0):.2f}")
                        if cols[3].button("Mark Paid", key=f"prof_paid_{r['ID']}"):
                            with SessionLocal() as db2:
                                b = db2.get(Booking, r["ID"])
                                b.paid = True; b.paid_at = datetime.now(timezone.utc)
                                db2.commit()
                            st.rerun()
                        if cols[4].button("Edit", key=f"prof_edit_{r['ID']}"):
                            start_edit_booking(r["ID"])


# ---------------- Bookings ----------------
def clear_booking_form():
    for k in [
        "bk_dog_id", "bk_type", "bk_status",
        "bk_sdate", "bk_stime", "bk_edate", "bk_etime",
        "bk_loc", "bk_notes",
        "bk_override_price", "bk_price",
        "bk_manual_disc_type", "bk_manual_disc_value", "bk_apply_sibling",
        "bk_paid",
        "bk_repeat_on", "bk_repeat_mode", "bk_repeat_every", "bk_repeat_until",
        "bk_edit_loaded_id", "bk_form_prefilled",
        "pending_switch_to_edit_mode", "pending_edit_booking_id"
    ]:
        st.session_state.pop(k, None)

def load_booking_into_form(booking_id: str):
    owner_id = current_user_id()
    if not owner_id:
        return
    with SessionLocal() as db:
        b = db.get(Booking, booking_id)
        if not b or b.owner_id != owner_id:
            return
        d = db.get(Dog, b.dog_id)
        tz_name = get_settings(db, owner_id).tz_name
        s_l = to_local(b.start_utc, tz_name)
        e_l = to_local(b.end_utc, tz_name)
        st.session_state.update({
            "bk_dog_id": d.id,
            "bk_type": b.service_type,
            "bk_status": b.status,
            "bk_sdate": s_l.date(),
            "bk_stime": s_l.time().replace(second=0, microsecond=0),
            "bk_edate": e_l.date(),
            "bk_etime": e_l.time().replace(second=0, microsecond=0),
            "bk_loc": b.location or "",
            "bk_notes": b.notes or "",
            "bk_paid": bool(b.paid),
            "bk_apply_sibling": bool(b.sibling_discount_applied),
            "bk_manual_disc_type": (b.discount_type or "None").capitalize()
            if (b.discount_type or "").lower() in ("percent", "amount") else "None",
            "bk_manual_disc_value": float(b.discount_value or 0.0),
            "bk_override_price": (b.discount_type or "").lower() == "override",
            "bk_price": float(b.price or 0.0),
            "bk_edit_loaded_id": b.id,
            "bk_form_prefilled": True,
        })

def start_edit_booking(booking_id: str):
    # Route via pending flags; radio not yet created in next run
    st.session_state["pending_switch_to_edit_mode"] = True
    st.session_state["pending_edit_booking_id"] = booking_id
    st.session_state.page = "Bookings"
    st.rerun()

def _handle_cap_errors(res, tz_name: str, db: Session):
    kind = res[0]
    if kind == "capacity":
        st.error(res[1])
        overlaps = res[2]
        rows = []; cache = {}
        for b in overlaps:
            if b.dog_id not in cache:
                cache[b.dog_id] = db.get(Dog, b.dog_id)
            rows.append({
                "Dog": cache[b.dog_id].name, "Service": b.service_type,
                "Start": to_local(b.start_utc, tz_name), "End": to_local(b.end_utc, tz_name),
                "Status": b.status, "Paid": bool(b.paid)
            })
        st.dataframe(pd.DataFrame(rows).sort_values("Start"), use_container_width=True)
    elif kind == "daily":
        st.error(res[1])

def bookings_section():
    if not current_user_id():
        auth_page(); return

    # --- Pre-hook BEFORE any widgets using 'booking_mode'
    if st.session_state.get("pending_switch_to_edit_mode"):
        st.session_state["booking_mode"] = "Add / Edit"
        st.session_state["bk_edit_loaded_id"] = st.session_state.get("pending_edit_booking_id")
        st.session_state["bk_form_prefilled"] = False
        st.session_state.pop("pending_switch_to_edit_mode", None)
        st.session_state.pop("pending_edit_booking_id", None)

    topbar()
    st.header("📅 Bookings")
    default_mode = st.session_state.get("booking_mode", "Add / Edit")
    mode = st.radio("Mode", ["Add / Edit", "Manager & Search"], horizontal=True, key="booking_mode",
                    index=["Add / Edit", "Manager & Search"].index(default_mode))

    owner_id = current_user_id()
    with SessionLocal() as db:
        settings = get_settings(db, owner_id)
        tz_name = settings.tz_name
        dogs = list(db.execute(select(Dog).where(Dog.owner_id == owner_id).order_by(Dog.name)).scalars())
        if not dogs:
            st.info("Add a dog first in **Doggy Profiles**.")
            return

        # Quick Edit
        df_all = bookings_df(db, owner_id, tz_name)
        if not df_all.empty:
            quick_label_map = {r["ID"]: f"{r['Dog']} — {r['Service']} — {r['Start (local)']:%d %b %H:%M}"
                               for _, r in df_all.sort_values("Start (local)").iterrows()}
            with st.expander("Quick Edit an existing booking", expanded=False):
                pick = st.selectbox("Pick a booking", options=["-- select --"] + list(quick_label_map.keys()),
                                    format_func=lambda k: quick_label_map.get(k, k) if k != "-- select --" else k,
                                    key="bk_quick_edit_select")
                if pick != "-- select --":
                    start_edit_booking(pick)

        loaded_id = st.session_state.get("bk_edit_loaded_id")
        if mode == "Add / Edit" and loaded_id and not st.session_state.get("bk_form_prefilled"):
            load_booking_into_form(loaded_id)

        if mode == "Add / Edit":
            left, right = st.columns([1.55, 0.45])
            with left:
                # Dog select by ID
                dogs_by_id = {d.id: d for d in dogs}
                dog_options = list(dogs_by_id.keys())
                if "bk_dog_id" not in st.session_state and dog_options:
                    st.session_state["bk_dog_id"] = dog_options[0]
                dog_id = st.selectbox(
                    "Dog *",
                    options=dog_options,
                    format_func=lambda _id: f"{dogs_by_id[_id].name}" +
                    (f" · {dogs_by_id[_id].household}" if dogs_by_id[_id].household else ""),
                    key="bk_dog_id",
                )
                dog_choice = dogs_by_id[dog_id]

                stype = st.selectbox("Service type *", SERVICE_TYPES,
                                     index=SERVICE_TYPES.index(st.session_state.get("bk_type", "daycare")),
                                     key="bk_type")
                status = st.selectbox("Status", ["booked", "pending", "cancelled"],
                                      index=["booked", "pending", "cancelled"].index(st.session_state.get("bk_status", "booked")),
                                      key="bk_status")

                today_local = datetime.now(tz=tz.gettz(tz_name)).date()
                s_date = st.date_input("Start date *", value=st.session_state.get("bk_sdate", today_local),
                                       format="DD/MM/YYYY", key="bk_sdate")
                s_time = st.time_input("Start time *", value=st.session_state.get("bk_stime", time(9, 0)), key="bk_stime")

                if stype == "overnight":
                    e_date = st.date_input("End date *", value=st.session_state.get("bk_edate", today_local),
                                           format="DD/MM/YYYY", key="bk_edate")
                    e_time = st.time_input("End time *", value=st.session_state.get("bk_etime", time(10, 0)), key="bk_etime")
                else:
                    e_date = s_date
                    e_time = st.time_input("End time * (same day)", value=st.session_state.get("bk_etime", time(17, 0)), key="bk_etime")

                if st.button("Apply default end time", key="bk_apply_default_end"):
                    mins = {
                        "walk": settings.dur_walk_min,
                        "daycare": settings.dur_daycare_min,
                        "overnight": settings.dur_overnight_min,
                        "home_visit": settings.dur_home_visit_min
                    }[stype]
                    start_local = datetime.combine(s_date, s_time)
                    end_local = start_local + timedelta(minutes=int(mins or 0))
                    st.session_state["bk_edate"] = end_local.date()
                    st.session_state["bk_etime"] = end_local.time().replace(second=0, microsecond=0)
                    st.rerun()

                location = st.text_input("Location", key="bk_loc")
                notes = st.text_area("Notes", key="bk_notes")

                st.subheader("Pricing & Discount")
                s_local_prev = datetime.combine(s_date, s_time)
                e_local_prev = datetime.combine(e_date, e_time)
                if stype != "overnight":
                    e_local_prev = datetime.combine(s_date, e_time)

                base_price = price_for_booking(dog_choice, stype, s_local_prev, e_local_prev)
                sib_ct = overlapping_sibling_count(db, owner_id, dog_choice,
                                                   to_utc(s_local_prev, tz_name), to_utc(e_local_prev, tz_name))
                apply_sibling = st.checkbox(
                    f"Apply sibling discount ({settings.sibling_discount_percent}% default)",
                    value=st.session_state.get("bk_apply_sibling", sib_ct > 0), key="bk_apply_sibling"
                )
                manual_disc_type = st.selectbox("Additional discount", ["None", "Percent", "Amount"],
                                                index=["None", "Percent", "Amount"].index(st.session_state.get("bk_manual_disc_type", "None")),
                                                key="bk_manual_disc_type")
                manual_disc_value = 0.0
                if manual_disc_type != "None":
                    manual_disc_value = st.number_input("Discount value", min_value=0.0, step=0.5,
                                                        value=float(st.session_state.get("bk_manual_disc_value", 0.0)),
                                                        key="bk_manual_disc_value")
                total_disc = 0.0
                if apply_sibling and settings.sibling_discount_percent > 0:
                    total_disc += round(base_price * (settings.sibling_discount_percent / 100.0), 2)
                if manual_disc_type == "Percent":
                    total_disc += round(base_price * ((float(manual_disc_value or 0.0)) / 100.0), 2)
                elif manual_disc_type == "Amount":
                    total_disc += round(float(manual_disc_value or 0.0), 2)
                final_price = max(round(base_price - total_disc, 2), 0.0)

                override = st.checkbox("Override final price", value=st.session_state.get("bk_override_price", False),
                                       key="bk_override_price")
                if override:
                    final_price = st.number_input("Final price (£)", min_value=0.0, step=0.5,
                                                  value=float(st.session_state.get("bk_price", final_price)), key="bk_price")
                pc = st.columns(3)
                pc[0].metric("Base", f"£{base_price:,.2f}")
                pc[1].metric("Discount", f"£{total_disc:,.2f}")
                pc[2].metric("Final", f"£{final_price:,.2f}")

                paid = st.checkbox("Mark as Paid", value=st.session_state.get("bk_paid", False), key="bk_paid")

                with st.expander("Repeat booking (optional)"):
                    repeat_mode = st.selectbox("Repeat", ["None", "Weekly", "Daily"],
                                               index=["None", "Weekly", "Daily"].index(st.session_state.get("bk_repeat_mode", "None")),
                                               key="bk_repeat_mode")
                    if repeat_mode != "None":
                        if repeat_mode == "Weekly":
                            every = st.number_input("Every N weeks", min_value=1, step=1,
                                                    value=int(st.session_state.get("bk_repeat_every", 1) or 1), key="bk_repeat_every")
                            weekdays_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
                            sel = st.multiselect("On days", list(weekdays_map.keys()),
                                                 default=st.session_state.get("bk_repeat_on", ["Mon"]), key="bk_repeat_on")
                            until = st.date_input("Repeat until (inclusive)",
                                                  value=st.session_state.get("bk_repeat_until", s_date), key="bk_repeat_until")
                        else:
                            every = st.number_input("Every N days", min_value=1, step=1,
                                                    value=int(st.session_state.get("bk_repeat_every", 1) or 1), key="bk_repeat_every")
                            st.session_state["bk_repeat_on"] = []
                            until = st.date_input("Repeat until (inclusive)",
                                                  value=st.session_state.get("bk_repeat_until", s_date), key="bk_repeat_until")

                a1, a2, a3, a4 = st.columns(4)
                save_btn = a1.button("Save booking", key="bk_save_btn", use_container_width=True)
                dup_btn = a2.button("Duplicate", key="bk_dup_btn", use_container_width=True, disabled=not bool(loaded_id))
                del_btn = a3.button("Delete", key="bk_delete_btn", use_container_width=True, disabled=not bool(loaded_id))
                clear_btn = a4.button("Clear form", key="bk_clear_btn", use_container_width=True)

            with right:
                st.info("- **Per-day limits** (Settings) enforced per service\n"
                        "- **Overnight** spans days and bills per 24h\n"
                        "- **Repeat** creates series; caps checked per instance")

            def _apply_discounts_for_store(base: float):
                mt = st.session_state.get("bk_manual_disc_type", "None")
                mv = float(st.session_state.get("bk_manual_disc_value", 0.0) or 0.0)
                sib_apply = bool(st.session_state.get("bk_apply_sibling", False))
                sib_pct = getattr(settings, "sibling_discount_percent", 0)
                total = 0.0
                if sib_apply and sib_pct > 0:
                    total += round(base * (sib_pct / 100.0), 2)
                if mt.lower() == "percent":
                    total += round(base * (mv / 100.0), 2)
                elif mt.lower() == "amount":
                    total += round(mv, 2)
                final = max(round(base - total, 2), 0.0)
                if st.session_state.get("bk_override_price", False):
                    return float(st.session_state.get("bk_price", final)), "override", 0.0, total, sib_apply
                return final, mt.lower(), mv, total, sib_apply

            def _repeat_days(mode: str, every: int, start_day: date, until_day: date, weekdays: List[int]) -> List[date]:
                days = []
                if mode == "weekly":
                    cur = start_day
                    while cur <= until_day:
                        for wd in weekdays:
                            d = cur + timedelta(days=(wd - cur.weekday()) % 7)
                            if start_day <= d <= until_day:
                                days.append(d)
                        cur += timedelta(weeks=every)
                    days = sorted(set(days))
                elif mode == "daily":
                    step = timedelta(days=max(1, every))
                    d = start_day
                    while d <= until_day:
                        days.append(d)
                        d += step
                return days

            def _store_one(dog: Dog, stype: str, sL: datetime, eL: datetime,
                           status: str, as_new: bool, edit: Optional[Booking]):
                sU = to_utc(sL, tz_name); eU = to_utc(eL, tz_name)
                base = price_for_booking(dog, stype, sL, eL)
                final, mt, mv, disc_amt, sib_applied = _apply_discounts_for_store(base)
                ok_conc, peak, cap, overlaps = check_capacity(db, owner_id, stype, sU, eU,
                                                              exclude_booking_id=(edit.id if (edit and not as_new) else None))
                if not ok_conc and status == "booked":
                    return ("capacity", f"Concurrent capacity exceeded (peak {peak} > cap {cap})", overlaps)
                ok_daily, ex = check_daily_limit(db, owner_id, stype, sU, eU, tz_name,
                                                 exclude_id=(edit.id if (edit and not as_new) else None))
                if not ok_daily and status == "booked":
                    msg = "Daily limit exceeded on:\n" + "\n".join(f"- {d.isoformat()}: {cnt}/{lim}" for d, (cnt, lim) in ex.items())
                    return ("daily", msg, ex)

                if edit and not as_new:
                    b = edit
                    b.dog_id = dog.id
                    b.service_type = stype
                    b.status = status
                    b.start_utc = sU
                    b.end_utc = eU
                    b.location = st.session_state.get("bk_loc") or None
                    b.price_before_discount = base
                    b.discount_type = mt
                    b.discount_value = mv
                    b.discount_amount = disc_amt
                    b.sibling_discount_applied = bool(sib_applied)
                    b.price = final
                    b.paid = bool(st.session_state.get("bk_paid", False))
                    b.paid_at = (datetime.now(timezone.utc) if b.paid else None)
                    b.notes = st.session_state.get("bk_notes") or None
                else:
                    b = Booking(
                        id=new_id(), owner_id=owner_id, dog_id=dog.id, service_type=stype, status=status,
                        start_utc=sU, end_utc=eU, location=st.session_state.get("bk_loc") or None,
                        price_before_discount=base, discount_type=mt, discount_value=mv, discount_amount=disc_amt,
                        sibling_discount_applied=bool(sib_applied), price=final,
                        paid=bool(st.session_state.get("bk_paid", False)),
                        paid_at=(datetime.now(timezone.utc) if st.session_state.get("bk_paid", False) else None),
                        notes=st.session_state.get("bk_notes") or None
                    )
                    db.add(b)
                db.commit()
                return ("ok",)

            # Save / Duplicate / Delete / Clear
            if save_btn or dup_btn:
                s_local = datetime.combine(s_date, s_time)
                e_local = datetime.combine(e_date, e_time)
                if stype != "overnight":
                    e_local = datetime.combine(s_date, e_time)
                if e_local <= s_local:
                    st.error("End must be after start.")
                else:
                    dur = e_local - s_local
                    current_edit = db.get(Booking, loaded_id) if loaded_id else None
                    if dup_btn and current_edit:
                        res = _store_one(dog_choice, stype, s_local, e_local, status, True, current_edit)
                        if res[0] == "ok":
                            st.success("Duplicated ✅"); clear_booking_form(); st.rerun()
                        else:
                            _handle_cap_errors(res, tz_name, db)
                    elif current_edit:
                        res = _store_one(dog_choice, stype, s_local, e_local, status, False, current_edit)
                        if res[0] == "ok":
                            st.success("Booking updated ✅"); clear_booking_form(); st.rerun()
                        else:
                            _handle_cap_errors(res, tz_name, db)
                    else:
                        rep = st.session_state.get("bk_repeat_mode", "None")
                        if rep == "None":
                            res = _store_one(dog_choice, stype, s_local, e_local, status, True, None)
                            if res[0] == "ok":
                                st.success("Booking added ✅"); clear_booking_form(); st.rerun()
                            else:
                                _handle_cap_errors(res, tz_name, db)
                        else:
                            if rep == "Weekly":
                                every = int(st.session_state.get("bk_repeat_every", 1))
                                wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
                                days = _repeat_days("weekly", every, s_date,
                                                    st.session_state.get("bk_repeat_until", s_date),
                                                    [wd_map[d] for d in (st.session_state.get("bk_repeat_on", ["Mon"]) or ["Mon"])])
                            else:
                                every = int(st.session_state.get("bk_repeat_every", 1))
                                days = _repeat_days("daily", every, s_date, st.session_state.get("bk_repeat_until", s_date), [])
                            created = 0; skipped = 0
                            for dday in days:
                                sL = datetime.combine(dday, s_time); eL = sL + dur
                                res = _store_one(dog_choice, stype, sL, eL, status, True, None)
                                if res[0] == "ok":
                                    created += 1
                                else:
                                    skipped += 1
                            if created > 0:
                                st.success(f"Created {created} booking(s) ✅")
                            if skipped > 0:
                                st.warning(f"Skipped {skipped} due to capacity/daily limits.")
                            clear_booking_form(); st.rerun()

            if del_btn and loaded_id:
                with SessionLocal() as db2:
                    b = db2.get(Booking, loaded_id)
                    if b and b.owner_id == owner_id:
                        db2.delete(b); db2.commit()
                st.warning("Booking deleted."); clear_booking_form(); st.rerun()

            if clear_btn:
                clear_booking_form(); st.rerun()

        else:
            cols = st.columns(6)
            start_day = cols[0].date_input("From", value=datetime.now().date() - timedelta(days=7))
            end_day = cols[1].date_input("To", value=datetime.now().date() + timedelta(days=21))
            dog_filter = cols[2].selectbox("Dog", ["All"] + [d.name for d in dogs])
            status_filter = cols[3].multiselect("Statuses", ["booked", "pending", "cancelled"], default=["booked", "pending"])
            paid_state = cols[4].selectbox("Paid", ["All", "Paid", "Unpaid"], index=0)
            qtext = cols[5].text_input("Search (dog/type/notes)")
            sU = to_utc(datetime.combine(start_day, time.min), tz_name)
            eU = to_utc(datetime.combine(end_day, time.max), tz_name)
            dog_id = None if dog_filter == "All" else next(d.id for d in dogs if d.name == dog_filter)
            paid_filter = {"All": None, "Paid": "paid", "Unpaid": "unpaid"}[paid_state]
            df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, dog_id=dog_id, statuses=status_filter or None, paid_filter=paid_filter)
            if not df.empty and qtext.strip():
                s = qtext.strip().lower()
                df = df[df.apply(lambda r: s in (r["Dog"] or "").lower()
                                  or s in (r["Service"] or "").lower()
                                  or s in ((r["Notes"] or "").lower()), axis=1)]
            if df.empty:
                st.info("No bookings for the selected filters.")
            else:
                st.dataframe(df.drop(columns=["DogID"]), use_container_width=True, height=360)
                st.markdown("#### Quick actions")
                for _, r in df.head(200).iterrows():
                    with st.container():
                        cols = st.columns([0.28, 0.30, 0.20, 0.10, 0.06, 0.06])
                        cols[0].write(f"**{r['Dog']}** — {r['Service']}")
                        cols[1].write(f"{r['Start (local)']:%d %b %H:%M} → {r['End (local)']:%d %b %H:%M}")
                        cols[2].write(f"£{(r['Price (£)'] or 0):.2f} • {'Paid' if r['Paid'] else 'Unpaid'}")
                        if cols[3].button("Edit", key=f"mgr_edit_{r['ID']}"):
                            start_edit_booking(r["ID"])
                        if cols[4].button(("💸" if not r["Paid"] else "↩︎"), key=f"mgr_pay_{r['ID']}"):
                            with SessionLocal() as db2:
                                b = db2.get(Booking, r["ID"])
                                if b and b.owner_id == owner_id:
                                    b.paid = not r["Paid"]
                                    b.paid_at = (datetime.now(timezone.utc) if b.paid else None)
                                    db2.commit()
                            st.rerun()
                        if cols[5].button("🗑️", key=f"mgr_del_{r['ID']}"):
                            with SessionLocal() as db2:
                                b = db2.get(Booking, r["ID"])
                                if b and b.owner_id == owner_id:
                                    db2.delete(b); db2.commit()
                            st.rerun()


# ---------------- Calendar ----------------
def _expand_into_days(df: pd.DataFrame, win_start: date, win_end: date) -> Dict[date, list]:
    by_day = {}
    if df.empty:
        return by_day
    for _, r in df.iterrows():
        s = r["Start (local)"]; e = r["End (local)"]
        s = max(s, datetime.combine(win_start, time.min, s.tzinfo))
        e = min(e, datetime.combine(win_end, time.max, s.tzinfo))
        d = s.date(); last = e.date()
        while d <= last:
            day_start = datetime.combine(d, time.min, s.tzinfo)
            day_end = datetime.combine(d, time.max, s.tzinfo)
            seg_s = max(s, day_start); seg_e = min(e, day_end)
            by_day.setdefault(d, []).append((r["Dog"], r["Service"], seg_s.time(), seg_e.time()))
            d += timedelta(days=1)
    return by_day

def _timeline_range_df(db: Session, owner_id: str, tz_name: str,
                       win_s_local: datetime, win_e_local: datetime,
                       types: List[str], show_pending: bool) -> pd.DataFrame:
    sU = to_utc(win_s_local, tz_name); eU = to_utc(win_e_local, tz_name)
    statuses = ["booked"] + (["pending"] if show_pending else [])
    df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, statuses=statuses)
    if df.empty:
        return df
    df = df[df["Service"].isin(types)]
    df["Start (local)"] = df["Start (local)"].apply(lambda s: max(s, win_s_local))
    df["End (local)"] = df["End (local)"].apply(lambda e: min(e, win_e_local))
    return df

def calendar_section():
    if not current_user_id():
        auth_page(); return
    topbar()
    st.header("🗓️ Calendar")
    owner_id = current_user_id()
    with SessionLocal() as db:
        tz_name = get_settings(db, owner_id).tz_name
        top = st.columns(5)
        view = top[0].selectbox("View", ["Month", "Week", "Day", "Timeline"], index=0, key="cal_view_select")
        types = top[1].multiselect("Booking types", SERVICE_TYPES, default=SERVICE_TYPES, key="cal_type_filter")
        group_by = top[2].selectbox("Group timeline by", ["Dog", "Service"], index=0, key="cal_group_by")
        show_pending = top[3].checkbox("Show pending too", value=False, key="cal_show_pending")
        if "cal_focus" not in st.session_state:
            st.session_state.cal_focus = datetime.now(tz=tz.gettz(tz_name)).date()
        if top[4].button("Today", key="cal_today_btn"):
            st.session_state.cal_focus = datetime.now(tz=tz.gettz(tz_name)).date(); st.rerun()

        if view == "Month":
            t = st.tabs(["Grid", "Timeline"])
            with t[0]:
                focus: date = st.session_state.get("cal_focus")
                hdr = st.columns(4)
                if hdr[0].button("◀︎ Prev", key="cal_grid_prev"):
                    first = focus.replace(day=1)
                    prev_end = first - timedelta(days=1)
                    st.session_state.cal_focus = prev_end.replace(day=1); st.rerun()
                # picker uses query_params-safe value
                _ = hdr[1].date_input("Month", value=focus, key="cal_month_picker", format="DD/MM/YYYY")
                if hdr[2].button("Next ▶︎", key="cal_grid_next"):
                    y = focus.year + (1 if focus.month == 12 else 0)
                    m = 1 if focus.month == 12 else focus.month + 1
                    st.session_state.cal_focus = date(y, m, 1); st.rerun()
                st.markdown(f"#### {focus.strftime('%B %Y')}")
                start_month = date(focus.year, focus.month, 1)
                _, last = pycal.monthrange(focus.year, focus.month)
                end_month = date(focus.year, focus.month, last)
                sU = to_utc(datetime.combine(start_month, time.min), tz_name)
                eU = to_utc(datetime.combine(end_month, time.max), tz_name)
                statuses = ["booked"] + (["pending"] if show_pending else [])
                df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, statuses=statuses)
                if not df.empty:
                    df = df[df["Service"].isin(types)]
                by_day = _expand_into_days(df, start_month, end_month) if not df.empty else {}
                cal = pycal.Calendar(firstweekday=0)
                grid = cal.monthdatescalendar(focus.year, focus.month)
                for week in grid:
                    cols = st.columns(7)
                    for i, d in enumerate(week):
                        in_m = (d.month == focus.month)
                        box = "background:#fff;border:1px solid #eee;border-radius:6px;padding:8px;height:160px;overflow:auto;"
                        if not in_m:
                            box = "background:#fafafa;border:1px dashed #eee;border-radius:6px;padding:8px;height:160px;opacity:0.75;overflow:auto;"
                        html = f"<div style='{box}'><div style='font-weight:600'>{d.day}</div>"
                        rows = by_day.get(d, [])
                        if rows:
                            for idx, (dog, service, t1, t2) in enumerate(rows[:6]):
                                html += f"<div style='font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>• {dog} ({service}) {t1.strftime('%H:%M')}–{t2.strftime('%H:%M')}</div>"
                            if len(rows) > 6:
                                html += f"<div style='font-size:12px;color:#888'>+{len(rows)-6} more…</div>"
                        else:
                            html += "<div style='font-size:12px;color:#aaa;margin-top:6px'>No bookings</div>"
                        html += "</div>"
                        cols[i].markdown(html, unsafe_allow_html=True)
            with t[1]:
                focus: date = st.session_state.get("cal_focus")
                hdr = st.columns(4)
                if hdr[0].button("◀︎ Prev", key="cal_tl_prev"):
                    first = focus.replace(day=1); prev_end = first - timedelta(days=1)
                    st.session_state.cal_focus = prev_end.replace(day=1); st.rerun()
                _ = hdr[1].date_input("Month", value=focus, key="cal_month_picker_t", format="DD/MM/YYYY")
                if hdr[2].button("Next ▶︎", key="cal_tl_next"):
                    y = focus.year + (1 if focus.month == 12 else 0)
                    m = 1 if focus.month == 12 else focus.month + 1
                    st.session_state.cal_focus = date(y, m, 1); st.rerun()
                st.markdown(f"#### {focus.strftime('%B %Y')} — Timeline")
                zone = tz.gettz(tz_name)
                win_s = datetime.combine(date(focus.year, focus.month, 1), time.min).replace(tzinfo=zone)
                last = pycal.monthrange(focus.year, focus.month)[1]
                win_e = datetime.combine(date(focus.year, focus.month, last), time.max).replace(tzinfo=zone)
                df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
                if df.empty:
                    st.info("No bookings this month.")
                else:
                    _timeline_plot(df, st.session_state.get("cal_group_by", "Dog"))
        elif view == "Week":
            focus: date = st.session_state.get("cal_focus")
            monday = focus - timedelta(days=focus.weekday())
            hdr = st.columns(4)
            if hdr[0].button("◀︎ Prev", key="cal_week_prev"):
                st.session_state.cal_focus = focus - timedelta(days=7); st.rerun()
            hdr[1].markdown(f"#### Week of {monday:%d %b %Y}")
            if hdr[2].button("Next ▶︎", key="cal_week_next"):
                st.session_state.cal_focus = focus + timedelta(days=7); st.rerun()
            zone = tz.gettz(tz_name)
            win_s = datetime.combine(monday, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(monday + timedelta(days=6), time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty:
                st.info("No bookings this week.")
            else:
                _timeline_plot(df, st.session_state.get("cal_group_by", "Dog"))
        elif view == "Day":
            focus: date = st.session_state.get("cal_focus")
            hdr = st.columns(4)
            if hdr[0].button("◀︎ Prev", key="cal_day_prev"):
                st.session_state.cal_focus = focus - timedelta(days=1); st.rerun()
            hdr[1].markdown(f"#### {focus:%A %d %B %Y}")
            if hdr[2].button("Next ▶︎", key="cal_day_next"):
                st.session_state.cal_focus = focus + timedelta(days=1); st.rerun()
            zone = tz.gettz(tz_name)
            win_s = datetime.combine(focus, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(focus, time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty:
                st.info("No bookings today.")
            else:
                _timeline_plot(df, st.session_state.get("cal_group_by", "Dog"))
        else:
            cols = st.columns(4)
            start_day = cols[0].date_input("From", value=datetime.now().date(), key="cal_free_from")
            end_day = cols[1].date_input("To", value=datetime.now().date() + timedelta(days=14), key="cal_free_to")
            zone = tz.gettz(tz_name)
            win_s = datetime.combine(start_day, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(end_day, time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty:
                st.info("No bookings in this range.")
            else:
                _timeline_plot(df, st.session_state.get("cal_group_by", "Dog"))


# ---------------- Export (.ics) ----------------
def build_ics(db: Session, owner_id: str, tz_name: str, sU: datetime, eU: datetime,
              statuses: List[str], alarm: int) -> bytes:
    cal = Calendar()
    cal.add("prodid", "-//Doggy Diary//EN")
    cal.add("version", "2.0")
    cal.add("X-WR-CALNAME", "Doggy Diary")
    cal.add("X-WR-TIMEZONE", tz_name)
    q = select(Booking, Dog).join(Dog, Dog.id == Booking.dog_id).where(
        Booking.owner_id == owner_id, Booking.start_utc < eU, Booking.end_utc > sU, Booking.status.in_(tuple(statuses))
    )
    for b, d in [(b, d) for b, d in db.execute(q).all()]:
        ev = Event()
        s = to_local(b.start_utc, tz_name); e = to_local(b.end_utc, tz_name)
        ev.add("uid", f"{b.id}@doggydiary")
        ev.add("summary", f"{b.service_type.title()}: {d.name}")
        desc = []
        if b.location:
            desc.append(f"Location: {b.location}")
        if b.price is not None:
            desc.append(f"Price: £{b.price:.2f}")
        if b.notes:
            desc.append(f"Notes: {b.notes}")
        if d.meds_notes:
            desc.append(f"Medications: {d.meds_notes}")
        if d.diet_notes:
            desc.append(f"Diet: {d.diet_notes}")
        ev.add("description", "\n".join(desc))
        ev.add("dtstart", s); ev.add("dtend", e)
        ev.add("dtstamp", datetime.now(timezone.utc))
        ev.add("categories", [b.service_type.upper()])
        try:
            alarm_ev = Alarm(); alarm_ev.add("action", "DISPLAY")
            alarm_ev.add("trigger", timedelta(minutes=-int(alarm)))
            alarm_ev.add("description", f"{b.service_type.title()} for {d.name}")
            ev.add_component(alarm_ev)
        except Exception:
            pass
        cal.add_component(ev)
    return cal.to_ical()

def export_section():
    if not current_user_id():
        auth_page(); return
    topbar()
    st.header("📤 Export to Calendar (.ics)")
    owner_id = current_user_id()
    with SessionLocal() as db:
        tz_name = get_settings(db, owner_id).tz_name
        col = st.columns(3)
        start_day = col[0].date_input("From", value=datetime.now().date(), key="ics_from")
        end_day = col[1].date_input("To", value=(datetime.now().date() + timedelta(days=14)), key="ics_to")
        include_status = col[2].multiselect("Statuses", ["booked", "pending"], default=["booked"], key="ics_statuses")
        sU = to_utc(datetime.combine(start_day, time.min), tz_name)
        eU = to_utc(datetime.combine(end_day, time.max), tz_name)
        if st.button("Generate .ics", key="ics_generate_btn", type="primary"):
            ics_bytes = build_ics(db, owner_id, tz_name, sU, eU, include_status, get_settings(db, owner_id).alarm_minutes)
            fname = f"doggy_diary_{start_day.isoformat()}_{end_day.isoformat()}.ics"
            st.download_button("Download .ics", key="ics_download_btn", data=ics_bytes,
                               file_name=fname, mime="text/calendar", use_container_width=True)
            st.info("Tip: Email to yourself and open on your phone to import. For a live subscription later, you’ll need a hosted feed.")


# ---------------- Settings ----------------
def settings_section():
    if not current_user_id():
        auth_page(); return
    topbar()
    st.header("⚙️ Settings")
    owner_id = current_user_id()
    with SessionLocal() as db:
        s = get_settings(db, owner_id)
        tz_input = st.text_input("IANA timezone (e.g., Europe/London)", value=s.tz_name, key="set_tz")
        alarm = st.number_input("Default reminder minutes (.ics)", min_value=0, max_value=240, step=5,
                                value=s.alarm_minutes, key="set_alarm")
        sib = st.number_input("Sibling discount (%)", min_value=0, max_value=100, step=1,
                              value=s.sibling_discount_percent, key="set_sib")
        st.subheader("Default durations")
        d1, d2, d3, d4 = st.columns(4)
        dur_walk = d1.number_input("Walk (min)", min_value=5, max_value=600, step=5,
                                   value=int(s.dur_walk_min or 60), key="set_dur_walk")
        dur_day = d2.number_input("Daycare (min)", min_value=60, max_value=24 * 60, step=15,
                                  value=int(s.dur_daycare_min or 480), key="set_dur_day")
        dur_over = d3.number_input("Overnight (min)", min_value=60, max_value=7 * 24 * 60, step=60,
                                   value=int(s.dur_overnight_min or 1440), key="set_dur_over")
        dur_home = d4.number_input("Home visit (min)", min_value=5, max_value=600, step=5,
                                   value=int(s.dur_home_visit_min or 60), key="set_dur_home")

        st.subheader("Concurrent capacity (max at the same time)")
        caps = list(db.execute(select(Capacity).where(Capacity.owner_id == owner_id)).scalars())
        cols = st.columns(len(SERVICE_TYPES)); conc_vals = {}
        for i, stype in enumerate(SERVICE_TYPES):
            cur = next((c.max_dogs for c in caps if c.service_type == stype), DEFAULT_CAPACITY[stype])
            conc_vals[stype] = cols[i].number_input(stype, min_value=0, max_value=50, step=1, value=cur, key=f"cap_{stype}")

        st.subheader("Daily limits per day (0 = unlimited)")
        dcap_rows = {c.service_type: c for c in db.execute(select(DailyCap).where(DailyCap.owner_id == owner_id)).scalars()}
        cols2 = st.columns(len(SERVICE_TYPES)); daily_vals = {}
        for i, stype in enumerate(SERVICE_TYPES):
            cur = dcap_rows.get(stype).max_per_day if stype in dcap_rows else 0
            daily_vals[stype] = cols2[i].number_input(stype, min_value=0, max_value=200, step=1, value=int(cur or 0), key=f"daily_{stype}")

        if st.button("Save settings", key="set_save_btn", type="primary"):
            s.tz_name = tz_input.strip() or s.tz_name
            s.alarm_minutes = int(alarm)
            s.sibling_discount_percent = int(sib)
            s.dur_walk_min = int(dur_walk)
            s.dur_daycare_min = int(dur_day)
            s.dur_overnight_min = int(dur_over)
            s.dur_home_visit_min = int(dur_home)

            existing = {c.service_type: c for c in caps}
            for stype, val in conc_vals.items():
                if stype in existing:
                    existing[stype].max_dogs = int(val)
                else:
                    db.add(Capacity(owner_id=owner_id, service_type=stype, max_dogs=int(val)))

            existing_daily = {c.service_type: c for c in db.execute(select(DailyCap).where(DailyCap.owner_id == owner_id)).scalars()}
            for stype, val in daily_vals.items():
                if stype in existing_daily:
                    existing_daily[stype].max_per_day = int(val or 0)
                else:
                    db.add(DailyCap(owner_id=owner_id, service_type=stype, max_per_day=int(val or 0)))
            db.commit(); st.success("Settings saved ✅")


# ---------------- Auth pages (improved) ----------------
def auth_page():
    st.title("🐶 Doggy Diary")
    st.subheader("Welcome back")
    tab_login, tab_reg = st.tabs(["Sign in", "Create account"])

    # Capture reset token from URL and route directly to reset page (new API)
    params = st.query_params
    token_from_url = None
    if "reset_token" in params:
        v = params.get("reset_token")
        token_from_url = v[0] if isinstance(v, list) else v
    if token_from_url:
        st.session_state["reset_token"] = token_from_url
        st.session_state.page = "Reset Password"
        st.query_params.clear()  # clear params
        st.rerun()

    with SessionLocal() as db:
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                pw = st.text_input("Password", type="password")
                c1, c2 = st.columns([0.5, 0.5])
                ok = c1.form_submit_button("Sign in", type="primary", use_container_width=True)
                if c2.form_submit_button("Forgot password?", use_container_width=True):
                    st.session_state.page = "Recover Account"; st.rerun()
            l2 = st.columns([0.5, 0.5])
            if l2[0].button("Contact us", help="Get help from our team", use_container_width=True, key="login_contact"):
                st.session_state.page = "Contact Us"; st.rerun()

            if ok:
                # Admin backdoor (username 'admin', password 'AdminPassword')
                if email.strip().lower() == "admin" and pw == "AdminPassword":
                    st.session_state["is_admin"] = True
                    st.session_state["user_id"] = None
                    st.session_state["user_name"] = "Admin"
                    st.session_state.page = "Admin"
                    st.rerun()

                u = db.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
                if not u or not verify_pwd(pw, u.password_hash):
                    st.error("Invalid email or password.")
                else:
                    st.session_state["user_id"] = u.id
                    st.session_state["user_name"] = u.full_name or u.email
                    ensure_user_defaults(u.id, u.email, db=db)
                    adopt_legacy_rows(u.id, db=db)
                    st.rerun()

        with tab_reg:
            with st.form("reg_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email (login)")
                pw1 = st.text_input("Password", type="password")
                pw2 = st.text_input("Confirm password", type="password")
                ok = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if ok:
                if not email.strip() or "@" not in email:
                    st.error("Enter a valid email."); return
                if pw1 != pw2 or len(pw1) < 6:
                    st.error("Passwords must match and be at least 6 chars."); return
                with SessionLocal() as db2:
                    exists = db2.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
                    if exists:
                        st.error("Email already registered."); return
                    u = User(id=new_id(), email=email.strip().lower(), full_name=name.strip() or None, password_hash=hash_pwd(pw1))
                    db2.add(u); db2.commit()
                st.success("Account created! Please sign in.")
                st.query_params.clear()  # clear any stray params


def recover_account_page():
    st.title("🔐 Recover your account")
    st.write("Enter your account email. We’ll send a reset link (and a one-time code).")
    with SessionLocal() as db:
        with st.form("recover_form"):
            email = st.text_input("Your account email")
            send_btn = st.form_submit_button("Send reset link", type="primary")
        if send_btn:
            u = db.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
            if not u:
                st.success("If an account exists for that email, a reset link will be sent.")
                return
            token = new_id()
            code = f"{random.randint(0, 999999):06d}"
            expires = datetime.now(timezone.utc) + timedelta(hours=2)
            pr = PasswordReset(user_id=u.id, token=token, code=code, expires_at=expires)
            db.add(pr); db.commit()

            link = None
            if APP_BASE_URL:
                link = f"{APP_BASE_URL.rstrip('/')}/?reset_token={token}"

            body_lines = [
                "Hi,",
                "",
                "We received a request to reset your Doggy Diary password.",
                f"One-time code: {code}",
            ]
            if link:
                body_lines += ["", f"Reset link: {link}", "", "This link expires in 2 hours."]
            else:
                body_lines += ["", "Open the app and choose 'Reset Password', then enter this code.", "This code expires in 2 hours."]
            body_lines += ["", "If you didn’t request this, you can ignore this email."]
            sent = send_email(u.email, "Reset your Doggy Diary password", "\n".join(body_lines))

            if sent:
                st.success("Check your email for the reset link/code.")
            else:
                st.warning("Email isn’t configured here. Use the one-time code below to reset now.")
                with st.expander("Reset now with one-time code", expanded=True):
                    st.code(code)
                    st.session_state["pending_reset_email"] = u.email
                    if st.button("Continue to Reset Password", type="primary"):
                        st.session_state.page = "Reset Password"; st.rerun()

    if st.button("Back to Sign in"):
        st.session_state.page = "Home"; st.rerun()


def reset_password_page():
    st.title("🔑 Reset password")
    with SessionLocal() as db:
        token = st.session_state.get("reset_token")
        email_prefill = st.session_state.get("pending_reset_email", "")
        tab_link, tab_code = st.tabs(["Use link", "Use one-time code"])

        with tab_link:
            st.caption("If you clicked a link from your email, it should appear here automatically.")
            tcol = st.columns([0.6, 0.4])
            token_in = tcol[0].text_input("Reset token", value=token or "", help="If empty, paste the token or use the code tab.")
            if tcol[1].button("Load", use_container_width=True):
                st.session_state["reset_token"] = token_in.strip()
                st.rerun()
            if token_in:
                pr = db.execute(select(PasswordReset).where(PasswordReset.token == token_in)).scalar_one_or_none()
                if not pr or pr.used_at or (pr.expires_at and pr.expires_at < datetime.now(timezone.utc)):
                    st.error("This reset token is invalid or expired.")
                else:
                    u = db.execute(select(User).where(User.id == pr.user_id)).scalar_one_or_none()
                    st.success(f"Resetting password for **{u.email}**")
                    with st.form("reset_form_link"):
                        p1 = st.text_input("New password", type="password")
                        p2 = st.text_input("Confirm new password", type="password")
                        ok = st.form_submit_button("Set new password", type="primary")
                    if ok:
                        if p1 != p2 or len(p1) < 6:
                            st.error("Passwords must match and be at least 6 chars.")
                        else:
                            u.password_hash = hash_pwd(p1)
                            pr.used_at = datetime.now(timezone.utc)
                            db.commit()
                            st.success("Password updated. You can sign in now.")
                            st.session_state.pop("reset_token", None)

        with tab_code:
            with st.form("reset_form_code"):
                email = st.text_input("Account email", value=email_prefill)
                code = st.text_input("One-time code")
                p1 = st.text_input("New password", type="password")
                p2 = st.text_input("Confirm new password", type="password")
                ok = st.form_submit_button("Set new password", type="primary")
            if ok:
                u = db.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
                if not u:
                    st.error("No account for that email.")
                else:
                    pr = db.execute(select(PasswordReset)
                                    .where(PasswordReset.user_id == u.id)
                                    .order_by(PasswordReset.id.desc())).scalars().first()
                    if not pr or pr.code != code or pr.used_at or (pr.expires_at and pr.expires_at < datetime.now(timezone.utc)):
                        st.error("Invalid or expired code.")
                    else:
                        if p1 != p2 or len(p1) < 6:
                            st.error("Passwords must match and be at least 6 chars.")
                        else:
                            u.password_hash = hash_pwd(p1)
                            pr.used_at = datetime.now(timezone.utc)
                            db.commit()
                            st.success("Password updated. You can sign in now.")
                            st.session_state.pop("pending_reset_email", None)

    if st.button("Back to Sign in"):
        st.session_state.page = "Home"; st.rerun()


# ---------------- Contact Us ----------------
def contact_page():
    st.title("💬 Contact us")
    st.write("Questions, issues, or feedback? Send us a message and we’ll get back to you.")
    with SessionLocal() as db:
        if current_user_id():
            u = db.execute(select(User).where(User.id == current_user_id())).scalar_one_or_none()
            from_email_default = u.email if u else ""
        else:
            from_email_default = ""

    with st.form("contact_form"):
        from_email = st.text_input("Your email", value=from_email_default)
        subject = st.text_input("Subject")
        message = st.text_area("Message", height=160)
        ok = st.form_submit_button("Send message", type="primary")
    if ok:
        if not from_email.strip() or "@" not in from_email:
            st.error("Please enter a valid email so we can reply.")
        elif not subject.strip() or not message.strip():
            st.error("Please add a subject and a message.")
        else:
            body = f"From: {from_email}\n\n{message}"
            sent = send_email(SUPPORT_TO, f"[Doggy Diary] {subject.strip()}", body)
            if sent:
                st.success("Message sent. Thanks for reaching out!")
            else:
                # Fallback: save to local inbox file
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                inbox = DATA_DIR / "support_inbox.csv"
                line = f'"{ts}","{from_email.replace("\"","\'")}","{subject.replace("\"","\'")}","{message.replace("\"","\'").replace("\n","\\n")}"\n'
                if not inbox.exists():
                    inbox.write_text('timestamp,from_email,subject,message\n', encoding="utf-8")
                with open(inbox, "a", encoding="utf-8") as f:
                    f.write(line)
                st.info("Message saved for review. (Email delivery is not configured on this server.)")

    if st.button("Back"):
        st.session_state.page = "Home"; st.rerun()


# ---------------- Admin ----------------
def admin_page():
    if not is_admin():
        st.warning("Admin access only."); return

    topbar()
    st.title("🛡️ Admin Dashboard")

    with SessionLocal() as db:
        st.subheader("Overview")
        total_users = db.execute(select(func.count(User.id))).scalar_one() or 0
        total_dogs = db.execute(select(func.count(Dog.id))).scalar_one() or 0
        total_bookings = db.execute(select(func.count(Booking.id))).scalar_one() or 0

        c = st.columns(3)
        c[0].metric("Users", f"{total_users}")
        c[1].metric("Dogs", f"{total_dogs}")
        c[2].metric("Bookings", f"{total_bookings}")

        st.markdown("### Users")
        q = st.text_input("Search users by email", value="")
        users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
        if q.strip():
            users = [u for u in users if q.lower() in (u.email or "").lower()]

        for u in users:
            with st.container():
                cols = st.columns([0.3, 0.25, 0.25, 0.2])
                cols[0].write(f"**{u.email}**")
                # per-user quick stats
                u_dogs = db.execute(select(func.count(Dog.id)).where(Dog.owner_id == u.id)).scalar_one() or 0
                u_book = db.execute(select(func.count(Booking.id)).where(Booking.owner_id == u.id)).scalar_one() or 0
                cols[1].write(f"Dogs: {u_dogs}")
                cols[2].write(f"Bookings: {u_book}")
                if cols[3].button("Impersonate", key=f"imp_{u.id}"):
                    st.session_state["user_id"] = u.id
                    st.session_state["user_name"] = u.full_name or u.email
                    st.session_state["impersonated_by_admin"] = True
                    st.session_state.page = "Home"
                    st.rerun()

        st.markdown("---")
        st.subheader("Global Bookings (read-only here)")
        # quick window
        today = datetime.now().date()
        sU = datetime.combine(today - timedelta(days=7), time.min).replace(tzinfo=tz.gettz(DEFAULT_TZ)).astimezone(timezone.utc)
        eU = datetime.combine(today + timedelta(days=21), time.max).replace(tzinfo=tz.gettz(DEFAULT_TZ)).astimezone(timezone.utc)
        rows = []
        q = (select(Booking, Dog, User)
             .join(Dog, Dog.id == Booking.dog_id)
             .join(User, User.id == Booking.owner_id)
             .where(Booking.start_utc < eU, Booking.end_utc > sU)
             .order_by(Booking.start_utc.desc()))
        for b, d, u in db.execute(q).all():
            rows.append({
                "User": u.email, "Dog": d.name, "Service": b.service_type, "Status": b.status,
                "Paid": bool(b.paid), "Start": b.start_utc, "End": b.end_utc, "Price": b.price
            })
        gdf = pd.DataFrame(rows)
        if gdf.empty:
            st.info("No bookings in the global window.")
        else:
            st.dataframe(gdf, use_container_width=True, height=280)
            st.caption("To edit a specific booking/dog, impersonate the user above and use the standard screens.")


# ---------------- Sidebar & routing ----------------
def sidebar_nav():
    with st.sidebar:
        st.header("Doggy Diary")
        who = st.session_state.get('user_name','Not signed in')
        if is_admin() and st.session_state.get("impersonated_by_admin") and current_user_id():
            who = f"Admin (as {who})"
        st.caption(f"Signed in as **{who}**")
        if is_admin():
            if st.button("🛡️ Admin", key="sb_admin_btn", use_container_width=True):
                st.session_state.page = "Admin"; st.rerun()
        if st.button("👤 My Profile", key="sb_profile_btn", use_container_width=True):
            st.session_state.page = "My Profile"; st.rerun()
        if st.button("🏠 Home", key="sb_home_btn", use_container_width=True):
            st.session_state.page = "Home"; st.rerun()
        st.markdown("### Sections")
        if st.button("🐾 Doggy Profiles", key="sb_profiles_btn", use_container_width=True):
            st.session_state.page = "Doggy Profiles"; st.rerun()
        if st.button("📅 Bookings", key="sb_bookings_btn", use_container_width=True):
            st.session_state.page = "Bookings"; st.rerun()
        if st.button("🗓️ Calendar", key="sb_calendar_btn", use_container_width=True):
            st.session_state.page = "Calendar"; st.rerun()
        st.markdown("---"); st.caption("Less frequent")
        if st.button("⚙️ Settings", key="sb_settings_btn", use_container_width=True):
            st.session_state.page = "Settings"; st.rerun()
        if st.button("📤 Export (.ics)", key="sb_export_btn", use_container_width=True):
            st.session_state.page = "Export"; st.rerun()
        st.markdown("---")
        if st.button("💬 Contact us", key="sb_contact_btn", use_container_width=True):
            st.session_state.page = "Contact Us"; st.rerun()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    sidebar_nav()
    p = st.session_state.page
    if p == "Admin":
        admin_page()
    elif p == "Home":
        if not current_user_id() and not is_admin():
            auth_page()
        else:
            nav_home()
    elif p == "Recover Account":
        recover_account_page()
    elif p == "Reset Password":
        reset_password_page()
    elif p == "Contact Us":
        contact_page()
    elif p == "My Profile":
        if not current_user_id():
            auth_page()
        else:
            my_profile_section()
    elif p == "Doggy Profiles":
        dogs_section()
    elif p == "Bookings":
        bookings_section()
    elif p == "Calendar":
        calendar_section()
    elif p == "Settings":
        settings_section()
    elif p == "Export":
        export_section()
    else:
        nav_home()

if __name__ == "__main__":
    main()
