#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Doggy Diary — Streamlit app for dog hotels (multi-tenant)
# Profiles, bookings (recurring, discounts, paid), calendar (grid + timelines),
# pricing, capacity/daily limits, insights, export to .ics.

from __future__ import annotations

import calendar as pycal
import math
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
for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{(DATA_DIR / 'doggy_diary.db').as_posix()}"

DEFAULT_TZ = "Europe/London"
SERVICE_TYPES = ["walk", "daycare", "overnight", "home_visit"]
DEFAULT_CAPACITY = {"walk": 4, "daycare": 6, "overnight": 2, "home_visit": 3}

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


Base.metadata.create_all(bind=engine)


# ---------- Migrations (robust: composite uniques, dedupe, indexes) ----------
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
    if not _table_exists(conn, "capacities"):
        return False
    if not _has_col(conn, "capacities", "owner_id"):
        return True
    sql = _table_sql(conn, "capacities") or ""
    if "UNIQUE" in sql and "owner_id" not in sql and "service_type" in sql:
        return True
    for _, is_unique, cols in _unique_indexes(conn, "capacities"):
        if is_unique and cols == ["service_type"]:
            return True
    return False

def _needs_rebuild_daily_caps(conn) -> bool:
    if not _table_exists(conn, "daily_caps"):
        return True
    if not _has_col(conn, "daily_caps", "owner_id"):
        return True
    sql = _table_sql(conn, "daily_caps") or ""
    if "UNIQUE" in sql and "owner_id" not in sql and "service_type" in sql:
        return True
    for _, is_unique, cols in _unique_indexes(conn, "daily_caps"):
        if is_unique and cols == ["service_type"]:
            return True
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
        # 1) Ensure required columns
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

        # 2) Rebuild capacities/daily_caps with composite uniques
        if _needs_rebuild_capacities(conn):
            _rebuild_capacities(conn)
        if _needs_rebuild_daily_caps(conn):
            _rebuild_daily_caps(conn)

        # 3) De-dupe singleton tables (settings, owner_profile) and enforce one-per-owner
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

        # 4) Helpful indexes
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_owner ON bookings(owner_id)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_time ON bookings(start_utc, end_utc)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_dogs_owner ON dogs(owner_id)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_capacity_owner_service ON capacities(owner_id, service_type)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_dailycap_owner_service ON daily_caps(owner_id, service_type)"))

run_migrations()


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
        if current_user_id() and st.button("Log out", key="logout_btn"):
            for k in list(st.session_state.keys()):
                if k.startswith(("user_", "bk_", "cal_", "sb_", "home_", "prof_")) or k in ("page", "booking_mode"):
                    st.session_state.pop(k, None)
            st.session_state.pop("user_id", None)
            st.session_state.pop("user_name", None)
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
        default_end = date(this_year, 12, 31)  # full year, not just to today
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

        # Quick Edit: jump straight to edit a booking (min clicks)
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
                # Dog select by ID (no ORM objects in widgets)
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
                    # Ease-of-life: end date is implicitly same-day; show as info
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
                hdr[1].date_input("Month", value=focus, key="cal_month_picker", format="DD/MM/YYYY")
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
                hdr[1].date_input("Month", value=focus, key="cal_month_picker_t", format="DD/MM/YYYY")
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


# ---------------- Auth page ----------------
def auth_page():
    st.title("🐶 Doggy Diary")
    st.subheader("Sign in or create an account")
    tab_login, tab_reg = st.tabs(["Sign in", "Create account"])
    with SessionLocal() as db:
        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email")
                pw = st.text_input("Password", type="password")
                ok = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if ok:
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
                exists = db.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
                if exists:
                    st.error("Email already registered."); return
                u = User(id=new_id(), email=email.strip().lower(), full_name=name.strip() or None, password_hash=hash_pwd(pw1))
                db.add(u); db.commit()
                st.session_state["user_id"] = u.id
                st.session_state["user_name"] = u.full_name or u.email
                ensure_user_defaults(u.id, u.email, db=db)
                adopt_legacy_rows(u.id, db=db)
                st.success("Account created!")
                st.rerun()


# ---------------- Sidebar & main routing ----------------
def sidebar_nav():
    with st.sidebar:
        st.header("Doggy Diary")
        st.caption(f"Signed in as **{st.session_state.get('user_name','Not signed in')}**")
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

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    sidebar_nav()
    p = st.session_state.page
    if p == "Home":
        if not current_user_id():
            auth_page()
        else:
            nav_home()
    elif p == "My Profile":
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
