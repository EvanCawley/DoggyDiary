#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Doggy Diary — Streamlit app (multi-tenant)
# Includes: Auth + password reset + contact us; per-user data isolation;
# bookings with edit-from-list, recurring, discounts, sibling pricing, paid flag;
# calendar (grid + multi-day timeline); per-day & concurrent capacity; .ics export;
# 30-day free trial then £12/yr subscription; Admin (impersonate, edit expiry/comp, delete accounts).

from __future__ import annotations

import os, smtplib, random, math, calendar as pycal
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

def _secret(name, env_fallback=None):
    # Safe secrets access (doesn't crash if no secrets file)
    try:
        return st.secrets.get(name)
    except Exception:
        return os.environ.get(env_fallback or name.upper())

APP_BASE_URL = _secret("app_base_url", "APP_BASE_URL")
SUBSCRIBE_URL = _secret("subscribe_url", "SUBSCRIBE_URL")  # hosted checkout/payment link

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
    # subscription
    sub_expires_at = Column(DateTime(timezone=True))     # when access ends (trial or paid)
    sub_comped = Column(Boolean, default=False)          # lifetime/comped access

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
    service_type = Column(String, nullable=False)
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

# ---------- Migrations ----------
from sqlalchemy import text as _sql_text
def _table_exists(conn, name: str) -> bool:
    return conn.execute(_sql_text(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"
    ), {"n": name}).fetchone() is not None
def _has_col(conn, table: str, col: str) -> bool:
    return any(r[1] == col for r in conn.execute(_sql_text(f"PRAGMA table_info({table})")).fetchall())
def _unique_indexes(conn, table: str):
    out = []
    for row in conn.execute(_sql_text(f"PRAGMA index_list({table})")).fetchall():
        idx_name = row[1]; is_unique = bool(row[2])
        cols = [r[2] for r in conn.execute(_sql_text(f"PRAGMA index_info({idx_name})")).fetchall()]
        out.append((idx_name, is_unique, cols))
    return out
def _needs_rebuild_capacities(conn) -> bool:
    if not _table_exists(conn, "capacities"): return False
    if not _has_col(conn, "capacities", "owner_id"): return True
    for _, is_unique, cols in _unique_indexes(conn, "capacities"):
        if is_unique and cols == ["service_type"]: return True
    return False
def _needs_rebuild_daily_caps(conn) -> bool:
    if not _table_exists(conn, "daily_caps"): return True
    if not _has_col(conn, "daily_caps", "owner_id"): return True
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
        # per-owner columns
        for t in ["settings", "capacities", "daily_caps", "owner_profile", "dogs", "bookings"]:
            if _table_exists(conn, t) and not _has_col(conn, t, "owner_id"):
                conn.execute(_sql_text(f"ALTER TABLE {t} ADD COLUMN owner_id TEXT"))
        # settings extra cols
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
        # dogs extra cols
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
        # bookings extra cols + status rename
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
        # users: subscription cols
        if _table_exists(conn, "users"):
            if not _has_col(conn, "users", "sub_expires_at"):
                conn.execute(_sql_text("ALTER TABLE users ADD COLUMN sub_expires_at TIMESTAMP"))
            if not _has_col(conn, "users", "sub_comped"):
                conn.execute(_sql_text("ALTER TABLE users ADD COLUMN sub_comped INTEGER DEFAULT 0"))
        # rebuild uniqueness for caps tables if needed
        if _needs_rebuild_capacities(conn): _rebuild_capacities(conn)
        if _needs_rebuild_daily_caps(conn): _rebuild_daily_caps(conn)
        # singleton de-dupes
        conn.execute(_sql_text("""
            DELETE FROM settings
            WHERE owner_id IS NOT NULL
              AND id NOT IN (SELECT MAX(id) FROM settings WHERE owner_id IS NOT NULL GROUP BY owner_id)
        """))
        conn.execute(_sql_text("""
            DELETE FROM owner_profile
            WHERE owner_id IS NOT NULL
              AND id NOT IN (SELECT MAX(id) FROM owner_profile WHERE owner_id IS NOT NULL GROUP BY owner_id)
        """))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_settings_owner ON settings(owner_id)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_owner_profile_owner ON owner_profile(owner_id)"))
        # useful indexes
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_owner ON bookings(owner_id)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_bookings_time ON bookings(start_utc, end_utc)"))
        conn.execute(_sql_text("CREATE INDEX IF NOT EXISTS ix_dogs_owner ON dogs(owner_id)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_capacity_owner_service ON capacities(owner_id, service_type)"))
        conn.execute(_sql_text("CREATE UNIQUE INDEX IF NOT EXISTS ux_dailycap_owner_service ON daily_caps(owner_id, service_type)"))
        # password resets table
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
    try:
        if not _smtp_creds_ok():
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            (OUTBOX_DIR / f"mail-{ts}.txt").write_text(f"TO: {to_email}\nSUBJECT: {subject}\n\n{body}", encoding="utf-8")
            return False
        host = os.environ.get("SMTP_HOST"); port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ.get("SMTP_USER"); pwd = os.environ.get("SMTP_PASS")
        use_tls = os.environ.get("SMTP_TLS", "1") not in ("0", "false", "False")
        from_addr = os.environ.get("SMTP_FROM", user)
        msg = EmailMessage(); msg["From"]=from_addr; msg["To"]=to_email; msg["Subject"]=subject; msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=20) as s:
            if use_tls: s.starttls()
            s.login(user, pwd)
            s.send_message(msg)
        return True
    except Exception:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        (OUTBOX_DIR / f"mail-{ts}.txt").write_text(f"TO: {to_email}\nSUBJECT: {subject}\n\n{body}", encoding="utf-8")
        return False

# ---------------- Auth & subscriptions ----------------
def new_id() -> str:
    import uuid; return uuid.uuid4().hex

def hash_pwd(p: str) -> str: return bcrypt.hash(p)

def verify_pwd(p: str, h: str) -> bool:
    try: return bcrypt.verify(p, h)
    except Exception: return False

def current_user_id() -> Optional[str]: return st.session_state.get("user_id")
def is_admin() -> bool: return bool(st.session_state.get("is_admin"))

def ensure_trial_if_missing(db: Session, u: User):
    if u.sub_comped: return
    if not u.sub_expires_at:
        u.sub_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        db.commit()

def sub_status_tuple(u: User) -> tuple[str, Optional[int], Optional[datetime]]:
    if u.sub_comped:
        return ("comped", None, None)
    now = datetime.now(timezone.utc)
    if u.sub_expires_at and u.sub_expires_at >= now:
        days_left = (u.sub_expires_at.date() - now.date()).days
        if u.created_at and (now - u.created_at).days < 30:
            return ("trial", days_left, u.sub_expires_at)
        return ("active", days_left, u.sub_expires_at)
    return ("expired", None, u.sub_expires_at)

def is_sub_active_for(u: User) -> bool:
    if u.sub_comped: return True
    if not u.sub_expires_at: return False
    return u.sub_expires_at >= datetime.now(timezone.utc)

def entitlement_guard():
    """If a signed-in user is not entitled, route to paywall."""
    if is_admin() and st.session_state.get("impersonated_by_admin"):
        return  # admin override while impersonating
    uid = current_user_id()
    if not uid: return
    with SessionLocal() as db:
        u = db.get(User, uid)
        ensure_trial_if_missing(db, u)
        if not is_sub_active_for(u):
            st.session_state.page = "Subscribe"
            st.rerun()

def ensure_user_defaults(owner_id: str, email: str, *, db: Optional[Session] = None):
    close = False
    if db is None: db = SessionLocal(); close = True
    try:
        # SETTINGS singleton
        rows = db.execute(select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())).scalars().all()
        if rows:
            for extra in rows[1:]: db.delete(extra)
            if not rows[0].tz_name: rows[0].tz_name = DEFAULT_TZ
        else:
            db.add(AppSetting(owner_id=owner_id, tz_name=DEFAULT_TZ, alarm_minutes=15,
                              sibling_discount_percent=20, dur_walk_min=60, dur_daycare_min=480,
                              dur_overnight_min=1440, dur_home_visit_min=60))
        # OWNER PROFILE singleton
        rows_p = db.execute(select(OwnerProfile).where(OwnerProfile.owner_id == owner_id).order_by(OwnerProfile.id.desc())).scalars().all()
        if rows_p:
            for extra in rows_p[1:]: db.delete(extra)
        else:
            db.add(OwnerProfile(owner_id=owner_id, name="", email=email, phone=""))
        # Capacities & daily caps
        existing = {c.service_type: c for c in db.execute(select(Capacity).where(Capacity.owner_id == owner_id)).scalars()}
        for stype, cap in DEFAULT_CAPACITY.items():
            if stype not in existing: db.add(Capacity(owner_id=owner_id, service_type=stype, max_dogs=cap))
        existing_d = {c.service_type: c for c in db.execute(select(DailyCap).where(DailyCap.owner_id == owner_id)).scalars()}
        for stype in SERVICE_TYPES:
            if stype not in existing_d: db.add(DailyCap(owner_id=owner_id, service_type=stype, max_per_day=None))
        db.commit()
    finally:
        if close: db.close()

def adopt_legacy_rows(owner_id: str, *, db: Optional[Session] = None):
    close = False
    if db is None: db = SessionLocal(); close = True
    try:
        db.execute(text("""
            UPDATE capacities SET owner_id=:oid
            WHERE owner_id IS NULL AND service_type NOT IN (SELECT service_type FROM capacities WHERE owner_id=:oid)
        """), {"oid": owner_id})
        db.execute(text("DELETE FROM capacities WHERE owner_id IS NULL"))
        db.execute(text("""
            UPDATE daily_caps SET owner_id=:oid
            WHERE owner_id IS NULL AND service_type NOT IN (SELECT service_type FROM daily_caps WHERE owner_id=:oid)
        """), {"oid": owner_id})
        db.execute(text("DELETE FROM daily_caps WHERE owner_id IS NULL"))
        # singletons
        existing = db.execute(select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())).scalars().all()
        nulls = db.execute(select(AppSetting).where(AppSetting.owner_id.is_(None)).order_by(AppSetting.id.desc())).scalars().all()
        if existing: [db.delete(r) for r in nulls]
        elif nulls:
            keep = nulls[0]; keep.owner_id = owner_id
            [db.delete(r) for r in nulls[1:]]
        existing_p = db.execute(select(OwnerProfile).where(OwnerProfile.owner_id == owner_id).order_by(OwnerProfile.id.desc())).scalars().all()
        nulls_p = db.execute(select(OwnerProfile).where(OwnerProfile.owner_id.is_(None)).order_by(OwnerProfile.id.desc())).scalars().all()
        if existing_p: [db.delete(r) for r in nulls_p]
        elif nulls_p:
            keep = nulls_p[0]; keep.owner_id = owner_id
            [db.delete(r) for r in nulls_p[1:]]
        for table in ["dogs", "bookings"]:
            db.execute(text(f"UPDATE {table} SET owner_id=:oid WHERE owner_id IS NULL"), {"oid": owner_id})
        db.commit()
    finally:
        if close: db.close()

# ---------------- Helpers ----------------
def get_settings(db: Session, owner_id: str) -> AppSetting:
    rows = db.execute(select(AppSetting).where(AppSetting.owner_id == owner_id).order_by(AppSetting.id.desc())).scalars().all()
    if not rows:
        s = AppSetting(owner_id=owner_id, tz_name=DEFAULT_TZ, alarm_minutes=15,
                       sibling_discount_percent=20, dur_walk_min=60, dur_daycare_min=480,
                       dur_overnight_min=1440, dur_home_visit_min=60)
        db.add(s); db.commit(); return s
    primary = rows[0]
    for extra in rows[1:]: db.delete(extra)
    db.commit(); return primary

def _aware_utc(dt):
    if dt is None: return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def to_utc(dt_local: datetime, tz_name: str) -> datetime:
    if dt_local.tzinfo is None: dt_local = dt_local.replace(tzinfo=tz.gettz(tz_name))
    return dt_local.astimezone(timezone.utc)

def to_local(dt_utc: datetime, tz_name: str) -> datetime:
    return dt_utc.astimezone(tz.gettz(tz_name) or tz.gettz(DEFAULT_TZ))

def sweepline_max_concurrency(intervals: List[Tuple[datetime, datetime]]) -> int:
    pts = []
    for s,e in intervals: pts.append((_aware_utc(s),+1)); pts.append((_aware_utc(e),-1))
    pts.sort(key=lambda x: (x[0], x[1])); cur=0; mx=0
    for _,d in pts: cur+=d; mx=max(mx,cur)
    return mx

def check_capacity(db: Session, owner_id: str, service_type: str,
                   start_utc: datetime, end_utc: datetime,
                   exclude_booking_id: Optional[str] = None):
    cap_row = db.execute(select(Capacity).where(Capacity.owner_id==owner_id, Capacity.service_type==service_type)).scalar_one_or_none()
    if not cap_row:
        cap_row = Capacity(owner_id=owner_id, service_type=service_type, max_dogs=DEFAULT_CAPACITY.get(service_type,5))
        db.add(cap_row); db.commit()
    cap = cap_row.max_dogs
    q = select(Booking).where(
        Booking.owner_id==owner_id, Booking.service_type==service_type, Booking.status=="booked",
        Booking.start_utc < end_utc, Booking.end_utc > start_utc
    )
    if exclude_booking_id: q=q.where(Booking.id!=exclude_booking_id)
    overlaps = list(db.execute(q).scalars())
    peak = sweepline_max_concurrency([(b.start_utc,b.end_utc) for b in overlaps]+[(start_utc,end_utc)])
    return peak<=cap, peak, cap, overlaps

def days_covered_local(s_utc: datetime, e_utc: datetime, tz_name: str) -> List[date]:
    s=to_local(s_utc,tz_name); e=to_local(e_utc,tz_name); d=s.date(); last=e.date(); out=[]
    while d<=last: out.append(d); d+=timedelta(days=1)
    return out

def check_daily_limit(db: Session, owner_id: str, service: str,
                      s_utc: datetime, e_utc: datetime, tz_name: str,
                      exclude_id: Optional[str] = None):
    row = db.execute(select(DailyCap).where(DailyCap.owner_id==owner_id, DailyCap.service_type==service)).scalar_one_or_none()
    limit = row.max_per_day if row else None
    if not limit or limit<=0: return True, {}
    exceeded={}
    for d in days_covered_local(s_utc,e_utc,tz_name):
        zone=tz.gettz(tz_name); s_l=datetime.combine(d,time.min,zone); e_l=datetime.combine(d,time.max,zone)
        sU=s_l.astimezone(timezone.utc); eU=e_l.astimezone(timezone.utc)
        q=select(func.count(Booking.id)).where(
            Booking.owner_id==owner_id, Booking.service_type==service, Booking.status=="booked",
            Booking.start_utc<eU, Booking.end_utc>sU
        )
        if exclude_id: q=q.where(Booking.id!=exclude_id)
        n=db.execute(q).scalar_one() or 0
        if n+1>limit: exceeded[d]=(n+1,limit)
    return (len(exceeded)==0), exceeded

def price_for_booking(dog: 'Dog', service: str, s_local: datetime, e_local: datetime) -> float:
    if service=="overnight":
        base=dog.price_overnight or 0.0
        seconds=max((e_local-s_local).total_seconds(),0.0)
        blocks=max(1, math.ceil(seconds/(24*3600)))
        return round(base*blocks,2)
    return round({
        "walk": dog.price_walk or 0.0, "daycare": dog.price_daycare or 0.0, "home_visit": dog.price_home_visit or 0.0
    }.get(service,0.0),2)

def overlapping_sibling_count(db: Session, owner_id: str, dog: 'Dog', s_utc: datetime, e_utc: datetime) -> int:
    if not dog.household: return 0
    q=(select(Booking,Dog).join(Dog,Dog.id==Booking.dog_id)
       .where(Dog.owner_id==owner_id, Booking.owner_id==owner_id, Dog.household==dog.household,
              Booking.dog_id!=dog.id, Booking.status=="booked", Booking.start_utc<e_utc, Booking.end_utc>s_utc))
    return len(db.execute(q).all())

def bookings_df(db: Session, owner_id: str, tz_name: str,
                start: datetime|None=None, end: datetime|None=None,
                dog_id: str|None=None, service: str|None=None,
                statuses: List[str]|None=None, paid_filter: str|None=None) -> pd.DataFrame:
    q=select(Booking,Dog).join(Dog,Dog.id==Booking.dog_id).where(Booking.owner_id==owner_id, Dog.owner_id==owner_id)
    if statuses: q=q.where(Booking.status.in_(tuple(statuses)))
    if start and end: q=q.where(Booking.start_utc<end, Booking.end_utc>start)
    if dog_id: q=q.where(Booking.dog_id==dog_id)
    if service: q=q.where(Booking.service_type==service)
    if paid_filter=="paid": q=q.where(Booking.paid==True)
    if paid_filter=="unpaid": q=q.where((Booking.paid==False)|(Booking.paid.is_(None)))
    rows=[(b,d) for b,d in db.execute(q).all()]
    data=[{
        "ID": b.id, "Dog": d.name, "DogID": d.id, "Household": d.household, "Service": b.service_type,
        "Status": b.status, "Paid": bool(b.paid),
        "Start (local)": to_local(b.start_utc, tz_name), "End (local)": to_local(b.end_utc, tz_name),
        "Price (£)": b.price, "Discount (£)": b.discount_amount,
        "Location": b.location, "Notes": b.notes
    } for b,d in rows]
    df=pd.DataFrame(data)
    if not df.empty: df=df.sort_values(by=["Start (local)","Dog"]).reset_index(drop=True)
    return df

def _timeline_plot(df: pd.DataFrame, group_by: str)->None:
    plot_df=pd.DataFrame({
        "Start": df["Start (local)"], "Finish": df["End (local)"],
        "Service": df["Service"].str.title(), "Dog": df["Dog"],
        "Resource": df["Dog"] if group_by=="Dog" else df["Service"].str.title(),
        "Details": df["Location"].fillna(""),
    })
    fig=px.timeline(plot_df, x_start="Start", x_end="Finish", y="Resource",
                    color="Service", hover_data=["Dog","Service","Start","Finish","Details"])
    fig.update_yaxes(autorange="reversed"); fig.update_layout(height=560, margin=dict(l=10,r=10,b=10,t=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI: topbar & nav ----------------
def topbar():
    cols=st.columns([0.07,0.73,0.20])
    with cols[0]:
        if st.button("👤", key="topbar_profile_btn", help="My Profile"):
            st.session_state.page="My Profile"; st.rerun()
    with cols[2]:
        if is_admin() and st.session_state.get("impersonated_by_admin"):
            if st.button("Exit impersonation", key="exit_imp_btn"):
                st.session_state.pop("user_id",None)
                st.session_state.pop("user_name",None)
                st.session_state.pop("impersonated_by_admin",None)
                st.session_state.page="Admin"; st.rerun()
        if st.button("Log out", key="logout_btn"):
            keys=list(st.session_state.keys())
            for k in keys: st.session_state.pop(k,None)
            st.rerun()

def nav_home():
    topbar()
    st.title("🐶 Doggy Diary")
    st.caption("Manage dog profiles, bookings, and your calendar at a glance.")
    c1,c2,c3=st.columns(3)
    if c1.button("🦴  Doggy Profiles", key="home_profiles_btn", use_container_width=True, type="primary"):
        st.session_state.page="Doggy Profiles"; st.rerun()
    if c2.button("📅  Bookings", key="home_bookings_btn", use_container_width=True, type="primary"):
        st.session_state.page="Bookings"; st.rerun()
    if c3.button("🗓️  Calendar", key="home_calendar_btn", use_container_width=True, type="primary"):
        st.session_state.page="Calendar"; st.rerun()
    st.markdown("---")
    st.write("Tip: Set **daily limits** and **concurrent capacity** in **Settings**.")

# ---------------- Doggy Profiles ----------------
def dogs_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
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
                            Image.open(photo).save(save); d.photo_path = str(save)
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
                    if e_dob: d.dob = datetime.combine(e_dob, time.min, tz.gettz(tz_name))
                    d.weight_kg = float(e_weight) if e_weight else None
                    d.vet_name = e_vn.strip() or None; d.vet_phone = e_vp.strip() or None
                    d.meds_notes = e_m.strip() or None; d.diet_notes = e_di.strip() or None; d.general_notes = e_no.strip() or None
                    d.household = e_hh.strip() or None
                    d.price_walk = float(e_pw); d.price_daycare = float(e_pd); d.price_overnight = float(e_po); d.price_home_visit = float(e_ph)
                    if e_photo is not None:
                        ext = Path(e_photo.name).suffix.lower(); save = UPLOAD_DIR / f"{d.id}{ext}"
                        Image.open(e_photo).save(save); d.photo_path = str(save)
                    db.commit(); st.success("Saved ✅")
                if delete:
                    db.delete(d); db.commit(); st.warning("Dog deleted."); st.rerun()

# ---------------- My Profile & Insights ----------------
def my_profile_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
    topbar()
    st.header("👤 My Profile & Earnings")
    owner_id = current_user_id()
    with SessionLocal() as db:
        # subscription summary
        u = db.get(User, owner_id)
        ensure_trial_if_missing(db, u)
        status, days_left, exp_at = sub_status_tuple(u)
        box = st.columns([0.5, 0.5])
        if status=="comped":
            box[0].success("Subscription: Lifetime (comped)")
        elif status in ("trial","active"):
            label="Trial" if status=="trial" else "Active"
            box[0].info(f"Subscription: {label} • Expires {exp_at.date().isoformat()} ({days_left} days left)")
        else:
            box[0].warning("Subscription: Expired — please subscribe to continue after payment")
        if SUBSCRIBE_URL:
            box[1].markdown(f"[Subscribe £12/year]({SUBSCRIBE_URL})")
        else:
            box[1].caption("Set SUBSCRIBE_URL to enable one-click subscribe")

        owner = db.execute(select(OwnerProfile).where(OwnerProfile.owner_id == owner_id)).scalar_one_or_none()
        if not owner: owner = OwnerProfile(owner_id=owner_id, name="", email="", phone=""); db.add(owner); db.commit()
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
        this_year = datetime.now().year
        default_start = date(this_year, 1, 1)
        default_end = date(this_year, 12, 31)
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
def _dog_list(db: Session, owner_id: str) -> List[Dog]:
    return list(db.execute(select(Dog).where(Dog.owner_id==owner_id).order_by(Dog.name)).scalars())

def start_edit_booking(booking_id: str):
    # prepare editing state *before* any widgets render
    st.session_state["editing_booking_id"] = booking_id
    st.session_state["booking_tab"] = "Add / Edit"
    st.session_state.page = "Bookings"
    st.rerun()

def clear_booking_edit_state():
    for k in list(st.session_state.keys()):
        if k.startswith("bk_"):  # form widget keys
            st.session_state.pop(k, None)
    st.session_state.pop("editing_booking_id", None)

def _service_default_duration(settings: AppSetting, service: str) -> int:
    return {
        "walk": settings.dur_walk_min or 60,
        "daycare": settings.dur_daycare_min or 480,
        "overnight": settings.dur_overnight_min or 1440,
        "home_visit": settings.dur_home_visit_min or 60,
    }.get(service, 60)

def _calc_price_preview(db: Session, owner_id: str, dog: Dog, service: str,
                        s_local: datetime, e_local: datetime,
                        discount_type: str, discount_value: float, tz_name: str) -> tuple[float, float, float, bool]:
    """Returns (base, discount_amount, final, sibling_applied)"""
    base = price_for_booking(dog, service, s_local, e_local)
    disc_amount = 0.0
    sibling_applied = False
    sU = to_utc(s_local, tz_name); eU = to_utc(e_local, tz_name)
    if discount_type == "percent":
        disc_amount = round(base * (max(discount_value,0.0)/100.0), 2)
    elif discount_type == "amount":
        disc_amount = round(max(min(discount_value, base), 0.0), 2)
    elif discount_type == "override":
        final = round(max(discount_value, 0.0), 2)
        return base, round(max(base - final, 0.0), 2), final, False
    # sibling discount if none selected and overlapping sibling exists
    settings = get_settings(db, owner_id)
    if discount_type == "none" and dog.household:
        if overlapping_sibling_count(db, owner_id, dog, sU, eU) > 0:
            pct = max(int(settings.sibling_discount_percent or 0), 0)
            if pct > 0:
                sibling_applied = True
                disc_amount = round(base * pct / 100.0, 2)
    final = round(max(base - disc_amount, 0.0), 2)
    return base, disc_amount, final, sibling_applied

def bookings_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
    topbar()
    st.header("📅 Bookings")
    owner_id = current_user_id()
    with SessionLocal() as db:
        settings = get_settings(db, owner_id)
        tz_name = settings.tz_name
        dogs = _dog_list(db, owner_id)
        if "booking_tab" not in st.session_state: st.session_state["booking_tab"] = "Add / Edit"
        if "editing_booking_id" not in st.session_state: st.session_state["editing_booking_id"] = None

        tab_add, tab_manage = st.tabs(["Add / Edit", "Manage existing"])
        # ---------------- Manage existing ----------------
        with tab_manage:
            c = st.columns([0.22,0.22,0.18,0.18,0.20])
            start_day = c[0].date_input("From", value=(datetime.now().date() - timedelta(days=7)), key="mgr_from")
            end_day = c[1].date_input("To", value=(datetime.now().date() + timedelta(days=21)), key="mgr_to")
            statuses = c[2].multiselect("Statuses", ["booked","pending","cancelled"], default=["booked","pending"], key="mgr_status")
            paid_opt = c[3].selectbox("Paid", ["all","paid","unpaid"], index=0, key="mgr_paid")
            dog_filter = c[4].selectbox("Dog filter", ["All"] + [d.name for d in dogs], index=0, key="mgr_dog")
            sU = to_utc(datetime.combine(start_day, time.min), tz_name)
            eU = to_utc(datetime.combine(end_day, time.max), tz_name)
            dog_id = None if dog_filter=="All" else (next((d.id for d in dogs if d.name==dog_filter), None))
            df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, dog_id=dog_id,
                             statuses=statuses if statuses else None,
                             paid_filter=(None if paid_opt=="all" else paid_opt))
            if df.empty:
                st.info("No bookings in range.")
            else:
                st.dataframe(df[["Dog","Service","Status","Paid","Start (local)","End (local)","Price (£)","Location","Notes"]],
                             use_container_width=True, height=300)
                st.markdown("#### Quick actions")
                for i, r in df.iterrows():
                    cols = st.columns([0.24,0.30,0.22,0.10,0.14])
                    cols[0].write(f"**{r['Dog']}** — {r['Service']}")
                    cols[1].write(f"{r['Start (local)']:%d %b %H:%M} → {r['End (local)']:%d %b %H:%M}")
                    cols[2].write(f"£{(r['Price (£)'] or 0):.2f} • {'Paid' if r['Paid'] else 'Unpaid'}")
                    if cols[3].button("Edit", key=f"mgr_edit_{r['ID']}"): start_edit_booking(r["ID"])
                    if cols[4].button("Delete", key=f"mgr_del_{r['ID']}"):
                        with SessionLocal() as dbd:
                            b = dbd.get(Booking, r["ID"])
                            if b: dbd.delete(b); dbd.commit()
                        st.success("Deleted."); st.rerun()

        # ---------------- Add / Edit ----------------
        with tab_add:
            editing_id = st.session_state.get("editing_booking_id")
            b_edit = db.get(Booking, editing_id) if editing_id else None

            if not dogs:
                st.warning("Add a dog first in **Doggy Profiles**.")
                return

            # Precompute defaults BEFORE rendering widgets
            zone = tz.gettz(tz_name)
            now_local = datetime.now(zone).replace(second=0, microsecond=0)
            if b_edit:
                d_default = next((d for d in dogs if d.id==b_edit.dog_id), dogs[0])
                service_default = b_edit.service_type
                status_default = b_edit.status
                paid_default = bool(b_edit.paid)
                s_local_default = to_local(b_edit.start_utc, tz_name)
                e_local_default = to_local(b_edit.end_utc, tz_name)
                loc_default = b_edit.location or ""
                notes_default = b_edit.notes or ""
                disc_type_def = b_edit.discount_type or "none"
                disc_val_def = float(b_edit.discount_value or 0.0)
                rec_default = False
                rec_weeks = 1
            else:
                d_default = dogs[0]
                service_default = "walk"
                status_default = "booked"
                paid_default = False
                s_local_default = now_local + timedelta(minutes=15)
                dur_min = _service_default_duration(settings, service_default)
                e_local_default = s_local_default + timedelta(minutes=dur_min if service_default!="overnight" else 60)
                loc_default = ""; notes_default = ""
                disc_type_def = "none"; disc_val_def = 0.0
                rec_default = False; rec_weeks = 1

            with st.form("booking_form", clear_on_submit=False):
                row1 = st.columns([0.30,0.22,0.18,0.15,0.15])
                dog_choice = row1[0].selectbox("Dog *", options=dogs, index=dogs.index(d_default),
                                               format_func=lambda d: d.name, key="bk_dog")
                service = row1[1].selectbox("Service *", SERVICE_TYPES, index=SERVICE_TYPES.index(service_default), key="bk_service")
                status = row1[2].selectbox("Status", ["booked","pending","cancelled"], index=["booked","pending","cancelled"].index(status_default), key="bk_status")
                paid = row1[3].checkbox("Paid", value=paid_default, key="bk_paid")
                location = row1[4].text_input("Location", value=loc_default, key="bk_loc")

                row2 = st.columns([0.24,0.24,0.24,0.24])
                s_day = row2[0].date_input("Start date *", value=s_local_default.date(), key="bk_sday")
                s_time = row2[1].time_input("Start time *", value=s_local_default.time(), step=300, key="bk_stime")

                if service=="overnight":
                    e_day = row2[2].date_input("End date *", value=e_local_default.date(), key="bk_eday")
                    e_time = row2[3].time_input("End time *", value=e_local_default.time(), step=300, key="bk_etime")
                else:
                    # auto duration for non-overnight
                    e_day = s_day
                    dur_min = _service_default_duration(settings, service)
                    e_time = (datetime.combine(date.today(), s_time) + timedelta(minutes=dur_min)).time()
                    row2[2].markdown(f"**End** (auto): {e_day.strftime('%d %b')} {e_time.strftime('%H:%M')}")
                    row2[3].markdown("&nbsp;")

                row3 = st.columns([0.24,0.20,0.20,0.36])
                disc_type = row3[0].selectbox("Discount type",
                                              ["none","percent","amount","override"],
                                              index=["none","percent","amount","override"].index(disc_type_def), key="bk_disc_type")
                disc_val = row3[1].number_input("Discount value", min_value=0.0, step=0.5, value=float(disc_val_def), key="bk_disc_val")
                recurring = row3[2].checkbox("Repeat weekly", value=rec_default, key="bk_recurring")
                repeat_weeks = row3[3].number_input("Repeat for N weeks", min_value=1, max_value=52, step=1, value=int(rec_weeks), key="bk_recweeks")

                notes = st.text_area("Notes", value=notes_default, key="bk_notes")

                # price preview
                s_local = datetime.combine(s_day, s_time, tzinfo=zone)
                e_local = datetime.combine(e_day, e_time, tzinfo=zone)
                base, disc_amt, final_price, sib_applied = _calc_price_preview(db, owner_id, dog_choice, service, s_local, e_local, disc_type, disc_val, tz_name)
                st.info(f"Price: base £{base:.2f}  —  discount £{disc_amt:.2f}"
                        + (" (sibling applied)" if sib_applied else "")
                        + f"  ⇒  **£{final_price:.2f}**")

                cta = st.columns([0.25,0.25,0.25,0.25])
                save_btn = cta[0].form_submit_button("Save booking", type="primary", use_container_width=True)
                clear_btn = cta[1].form_submit_button("Clear form", use_container_width=True)

            if clear_btn:
                clear_booking_edit_state(); st.success("Form cleared."); st.rerun()

            if save_btn:
                # sanity
                if service=="overnight" and (datetime.combine(e_day,e_time) <= datetime.combine(s_day,s_time)):
                    st.error("End must be after start for overnight."); st.stop()

                sU = to_utc(s_local, tz_name); eU = to_utc(e_local, tz_name)
                # capacity checks
                ok_cap, peak, cap, _ = check_capacity(db, owner_id, service, sU, eU, exclude_booking_id=(b_edit.id if b_edit else None))
                if not ok_cap:
                    st.error(f"Capacity exceeded for {service}. Peak {peak}/{cap}."); st.stop()
                ok_day, exceeded = check_daily_limit(db, owner_id, service, sU, eU, tz_name, exclude_id=(b_edit.id if b_edit else None))
                if not ok_day:
                    days_txt = ", ".join([f"{d.isoformat()} ({n}/{lim})" for d,(n,lim) in exceeded.items()])
                    st.error(f"Daily limit exceeded on: {days_txt}"); st.stop()

                # price
                base, disc_amt, final_price, sib_applied = _calc_price_preview(db, owner_id, dog_choice, service, s_local, e_local, disc_type, disc_val, tz_name)

                def _save_one(startU, endU):
                    nonlocal b_edit
                    if b_edit:
                        b = b_edit
                        b.dog_id = dog_choice.id
                        b.service_type = service
                        b.status = status
                        b.paid = bool(paid)
                        b.paid_at = (datetime.now(timezone.utc) if paid else None)
                        b.start_utc = startU; b.end_utc = endU
                        b.location = (location or None)
                        b.notes = (notes or None)
                        b.price_before_discount = base
                        b.discount_type = disc_type
                        b.discount_value = float(disc_val or 0.0)
                        b.discount_amount = disc_amt
                        b.sibling_discount_applied = sib_applied
                        b.price = final_price
                    else:
                        b = Booking(
                            id=new_id(), owner_id=owner_id, dog_id=dog_choice.id,
                            service_type=service, status=status, paid=bool(paid),
                            paid_at=(datetime.now(timezone.utc) if paid else None),
                            start_utc=startU, end_utc=endU, location=(location or None),
                            notes=(notes or None), price_before_discount=base,
                            discount_type=disc_type, discount_value=float(disc_val or 0.0),
                            discount_amount=disc_amt, sibling_discount_applied=sib_applied,
                            price=final_price
                        )
                        db.add(b)
                    db.commit()
                    return b

                # save primary
                saved = _save_one(sU, eU)

                # recurring: weekly copies (only when creating new)
                if (not b_edit) and recurring and repeat_weeks>1:
                    for i in range(1, int(repeat_weeks)):
                        sU_i = sU + timedelta(days=7*i)
                        eU_i = eU + timedelta(days=7*i)
                        ok_cap_i, _, _, _ = check_capacity(db, owner_id, service, sU_i, eU_i)
                        ok_day_i, exceeded_i = check_daily_limit(db, owner_id, service, sU_i, eU_i, tz_name)
                        if ok_cap_i and ok_day_i:
                            _save_one(sU_i, eU_i)

                st.success("Saved ✅")
                clear_booking_edit_state()
                st.rerun()
# ---------------- Calendar ----------------
def _expand_into_days(df: pd.DataFrame, win_start: date, win_end: date) -> Dict[date, list]:
    by_day = {}
    if df.empty: return by_day
    for _, r in df.iterrows():
        s=r["Start (local)"]; e=r["End (local)"]
        s=max(s, datetime.combine(win_start, time.min, s.tzinfo))
        e=min(e, datetime.combine(win_end, time.max, s.tzinfo))
        d=s.date(); last=e.date()
        while d<=last:
            day_start=datetime.combine(d,time.min,s.tzinfo); day_end=datetime.combine(d,time.max,s.tzinfo)
            seg_s=max(s,day_start); seg_e=min(e,day_end)
            by_day.setdefault(d,[]).append((r["Dog"], r["Service"], seg_s.time(), seg_e.time()))
            d+=timedelta(days=1)
    return by_day

def _timeline_range_df(db: Session, owner_id: str, tz_name: str,
                       win_s_local: datetime, win_e_local: datetime,
                       types: List[str], show_pending: bool) -> pd.DataFrame:
    sU = to_utc(win_s_local, tz_name); eU = to_utc(win_e_local, tz_name)
    statuses = ["booked"] + (["pending"] if show_pending else [])
    df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, statuses=statuses)
    if df.empty: return df
    df = df[df["Service"].isin(types)]
    df["Start (local)"]=df["Start (local)"].apply(lambda s: max(s, win_s_local))
    df["End (local)"]=df["End (local)"].apply(lambda e: min(e, win_e_local))
    return df

def calendar_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
    topbar()
    st.header("🗓️ Calendar")
    owner_id = current_user_id()
    with SessionLocal() as db:
        tz_name = get_settings(db, owner_id).tz_name
        top = st.columns(5)
        view = top[0].selectbox("View", ["Month", "Week", "Day", "Timeline"], index=0, key="cal_view_select")
        types = top[1].multiselect("Booking types", SERVICE_TYPES, default=SERVICE_TYPES, key="cal_type_filter")
        _ = top[2].selectbox("Group timeline by", ["Dog", "Service"], index=0, key="cal_group_by")
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
                    first = focus.replace(day=1); prev_end = first - timedelta(days=1)
                    st.session_state.cal_focus = prev_end.replace(day=1); st.rerun()
                _ = hdr[1].date_input("Month", value=focus, key="cal_month_picker", format="DD/MM/YYYY")
                if hdr[2].button("Next ▶︎", key="cal_grid_next"):
                    y = focus.year + (1 if focus.month == 12 else 0); m = 1 if focus.month == 12 else focus.month + 1
                    st.session_state.cal_focus = date(y, m, 1); st.rerun()
                st.markdown(f"#### {focus.strftime('%B %Y')}")
                start_month = date(focus.year, focus.month, 1)
                _, last = pycal.monthrange(focus.year, focus.month)
                end_month = date(focus.year, focus.month, last)
                sU = to_utc(datetime.combine(start_month, time.min), tz_name)
                eU = to_utc(datetime.combine(end_month, time.max), tz_name)
                statuses = ["booked"] + (["pending"] if show_pending else [])
                df = bookings_df(db, owner_id, tz_name, start=sU, end=eU, statuses=statuses)
                if not df.empty: df = df[df["Service"].isin(types)]
                by_day = _expand_into_days(df, start_month, end_month) if not df.empty else {}
                cal = pycal.Calendar(firstweekday=0); grid = cal.monthdatescalendar(focus.year, focus.month)
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
                    y = focus.year + (1 if focus.month == 12 else 0); m = 1 if focus.month == 12 else focus.month + 1
                    st.session_state.cal_focus = date(y, m, 1); st.rerun()
                st.markdown(f"#### {focus.strftime('%B %Y')} — Timeline")
                zone = tz.gettz(tz_name)
                win_s = datetime.combine(date(focus.year, focus.month, 1), time.min).replace(tzinfo=zone)
                last = pycal.monthrange(focus.year, focus.month)[1]
                win_e = datetime.combine(date(focus.year, focus.month, last), time.max).replace(tzinfo=zone)
                df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
                if df.empty: st.info("No bookings this month.")
                else: _timeline_plot(df, st.session_state.get("cal_group_by","Dog"))
        elif view == "Week":
            focus: date = st.session_state.get("cal_focus"); monday = focus - timedelta(days=focus.weekday())
            hdr = st.columns(4)
            if hdr[0].button("◀︎ Prev", key="cal_week_prev"): st.session_state.cal_focus = focus - timedelta(days=7); st.rerun()
            hdr[1].markdown(f"#### Week of {monday:%d %b %Y}")
            if hdr[2].button("Next ▶︎", key="cal_week_next"): st.session_state.cal_focus = focus + timedelta(days=7); st.rerun()
            zone = tz.gettz(tz_name); win_s = datetime.combine(monday, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(monday + timedelta(days=6), time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty: st.info("No bookings this week.")
            else: _timeline_plot(df, st.session_state.get("cal_group_by","Dog"))
        elif view == "Day":
            focus: date = st.session_state.get("cal_focus")
            hdr = st.columns(4)
            if hdr[0].button("◀︎ Prev", key="cal_day_prev"): st.session_state.cal_focus = focus - timedelta(days=1); st.rerun()
            hdr[1].markdown(f"#### {focus:%A %d %B %Y}")
            if hdr[2].button("Next ▶︎", key="cal_day_next"): st.session_state.cal_focus = focus + timedelta(days=1); st.rerun()
            zone = tz.gettz(tz_name); win_s = datetime.combine(focus, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(focus, time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty: st.info("No bookings today.")
            else: _timeline_plot(df, st.session_state.get("cal_group_by","Dog"))
        else:
            cols = st.columns(4)
            start_day = cols[0].date_input("From", value=datetime.now().date(), key="cal_free_from")
            end_day = cols[1].date_input("To", value=datetime.now().date() + timedelta(days=14), key="cal_free_to")
            zone = tz.gettz(tz_name); win_s = datetime.combine(start_day, time.min).replace(tzinfo=zone)
            win_e = datetime.combine(end_day, time.max).replace(tzinfo=zone)
            df = _timeline_range_df(db, owner_id, tz_name, win_s, win_e, types, show_pending)
            if df.empty: st.info("No bookings in this range.")
            else: _timeline_plot(df, st.session_state.get("cal_group_by","Dog"))

# ---------------- Export ----------------
def build_ics(db: Session, owner_id: str, tz_name: str, sU: datetime, eU: datetime,
              statuses: List[str], alarm: int) -> bytes:
    cal = Calendar(); cal.add("prodid","-//Doggy Diary//EN"); cal.add("version","2.0")
    cal.add("X-WR-CALNAME","Doggy Diary"); cal.add("X-WR-TIMEZONE", tz_name)
    q = select(Booking, Dog).join(Dog, Dog.id==Booking.dog_id).where(
        Booking.owner_id==owner_id, Booking.start_utc<eU, Booking.end_utc>sU, Booking.status.in_(tuple(statuses))
    )
    for b,d in [(b,d) for b,d in db.execute(q).all()]:
        ev=Event(); s=to_local(b.start_utc,tz_name); e=to_local(b.end_utc,tz_name)
        ev.add("uid", f"{b.id}@doggydiary"); ev.add("summary", f"{b.service_type.title()}: {d.name}")
        desc=[]
        if b.location: desc.append(f"Location: {b.location}")
        if b.price is not None: desc.append(f"Price: £{b.price:.2f}")
        if b.notes: desc.append(f"Notes: {b.notes}")
        if d.meds_notes: desc.append(f"Medications: {d.meds_notes}")
        if d.diet_notes: desc.append(f"Diet: {d.diet_notes}")
        ev.add("description","\n".join(desc)); ev.add("dtstart",s); ev.add("dtend",e); ev.add("dtstamp",datetime.now(timezone.utc))
        ev.add("categories",[b.service_type.upper()])
        try:
            alarm_ev=Alarm(); alarm_ev.add("action","DISPLAY"); alarm_ev.add("trigger", timedelta(minutes=-int(alarm)))
            alarm_ev.add("description", f"{b.service_type.title()} for {d.name}"); ev.add_component(alarm_ev)
        except Exception: pass
        cal.add_component(ev)
    return cal.to_ical()

def export_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
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
            st.info("Tip: Email to yourself and open on your phone to import.")

# ---------------- Settings ----------------
def settings_section():
    if not current_user_id(): auth_page(); return
    entitlement_guard()
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
                if stype in existing: existing[stype].max_dogs = int(val)
                else: db.add(Capacity(owner_id=owner_id, service_type=stype, max_dogs=int(val)))

            existing_daily = {c.service_type: c for c in db.execute(select(DailyCap).where(DailyCap.owner_id == owner_id)).scalars()}
            for stype, val in daily_vals.items():
                if stype in existing_daily: existing_daily[stype].max_per_day = int(val or 0)
                else: db.add(DailyCap(owner_id=owner_id, service_type=stype, max_per_day=int(val or 0)))
            db.commit(); st.success("Settings saved ✅")

# ---------------- Auth pages ----------------
def auth_page():
    st.title("🐶 Doggy Diary")
    st.subheader("Welcome back")
    tab_login, tab_reg = st.tabs(["Sign in", "Create account"])

    # Capture reset token from URL and route directly to reset page (new API)
    params = st.query_params
    token_from_url = None
    if "reset_token" in params:
        v = params.get("reset_token"); token_from_url = v[0] if isinstance(v, list) else v
    if token_from_url:
        st.session_state["reset_token"] = token_from_url
        st.session_state.page = "Reset Password"
        st.query_params.clear(); st.rerun()

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
                    st.session_state.page = "Admin"; st.rerun()

                u = db.execute(select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
                if not u or not verify_pwd(pw, u.password_hash):
                    st.error("Invalid email or password.")
                else:
                    st.session_state["user_id"] = u.id
                    st.session_state["user_name"] = u.full_name or u.email
                    ensure_user_defaults(u.id, u.email, db=db)
                    adopt_legacy_rows(u.id, db=db)
                    ensure_trial_if_missing(db, u)
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
                    u = User(id=new_id(), email=email.strip().lower(), full_name=name.strip() or None,
                             password_hash=hash_pwd(pw1),
                             sub_expires_at=datetime.now(timezone.utc) + timedelta(days=30),
                             sub_comped=False)
                    db2.add(u); db2.commit()
                st.success("Account created! You have **30 days free**. Please sign in.")
                st.query_params.clear()

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
            token = new_id(); code = f"{random.randint(0, 999999):06d}"
            expires = datetime.now(timezone.utc) + timedelta(hours=2)
            pr = PasswordReset(user_id=u.id, token=token, code=code, expires_at=expires)
            db.add(pr); db.commit()
            link = f"{APP_BASE_URL.rstrip('/')}/?reset_token={token}" if APP_BASE_URL else None
            body = ["Hi,","","We received a request to reset your Doggy Diary password.",f"One-time code: {code}"]
            if link: body += ["", f"Reset link: {link}", "", "This link expires in 2 hours."]
            else: body += ["", "Open the app and choose 'Reset Password', then enter this code.", "This code expires in 2 hours."]
            sent = send_email(u.email, "Reset your Doggy Diary password", "\n".join(body+["","","If you didn’t request this, ignore this email."]))
            if sent: st.success("Check your email for the reset link/code.")
            else:
                st.warning("Email isn’t configured here. Use the one-time code below to reset now.")
                with st.expander("Reset now with one-time code", expanded=True):
                    st.code(code)
                    st.session_state["pending_reset_email"] = u.email
                    if st.button("Continue to Reset Password", type="primary"):
                        st.session_state.page = "Reset Password"; st.rerun()
    if st.button("Back to Sign in"): st.session_state.page="Home"; st.rerun()

def reset_password_page():
    st.title("🔑 Reset password")
    with SessionLocal() as db:
        token = st.session_state.get("reset_token"); email_prefill = st.session_state.get("pending_reset_email","")
        tab_link, tab_code = st.tabs(["Use link", "Use one-time code"])
        with tab_link:
            st.caption("If you clicked a link from your email, it should appear here automatically.")
            tcol = st.columns([0.6,0.4])
            token_in = tcol[0].text_input("Reset token", value=token or "", help="If empty, paste the token or use the code tab.")
            if tcol[1].button("Load", use_container_width=True):
                st.session_state["reset_token"]=token_in.strip(); st.rerun()
            if token_in:
                pr = db.execute(select(PasswordReset).where(PasswordReset.token==token_in)).scalar_one_or_none()
                if not pr or pr.used_at or (pr.expires_at and pr.expires_at < datetime.now(timezone.utc)):
                    st.error("This reset token is invalid or expired.")
                else:
                    u = db.execute(select(User).where(User.id==pr.user_id)).scalar_one_or_none()
                    st.success(f"Resetting password for **{u.email}**")
                    with st.form("reset_form_link"):
                        p1 = st.text_input("New password", type="password")
                        p2 = st.text_input("Confirm new password", type="password")
                        ok = st.form_submit_button("Set new password", type="primary")
                    if ok:
                        if p1 != p2 or len(p1) < 6: st.error("Passwords must match and be at least 6 chars.")
                        else:
                            u.password_hash = hash_pwd(p1); pr.used_at = datetime.now(timezone.utc)
                            db.commit(); st.success("Password updated. You can sign in now."); st.session_state.pop("reset_token",None)
        with tab_code:
            with st.form("reset_form_code"):
                email = st.text_input("Account email", value=email_prefill)
                code = st.text_input("One-time code")
                p1 = st.text_input("New password", type="password")
                p2 = st.text_input("Confirm new password", type="password")
                ok = st.form_submit_button("Set new password", type="primary")
            if ok:
                u = db.execute(select(User).where(User.email==email.strip().lower())).scalar_one_or_none()
                if not u: st.error("No account for that email.")
                else:
                    pr = db.execute(select(PasswordReset).where(PasswordReset.user_id==u.id).order_by(PasswordReset.id.desc())).scalars().first()
                    if not pr or pr.code!=code or pr.used_at or (pr.expires_at and pr.expires_at<datetime.now(timezone.utc)):
                        st.error("Invalid or expired code.")
                    else:
                        if p1 != p2 or len(p1) < 6: st.error("Passwords must match and be at least 6 chars.")
                        else:
                            u.password_hash = hash_pwd(p1); pr.used_at = datetime.now(timezone.utc)
                            db.commit(); st.success("Password updated. You can sign in now."); st.session_state.pop("pending_reset_email",None)
    if st.button("Back to Sign in"): st.session_state.page="Home"; st.rerun()

# ---------------- Contact Us ----------------
def contact_page():
    st.title("💬 Contact us")
    st.write("Questions, issues, or feedback? Send us a message and we’ll get back to you.")
    with SessionLocal() as db:
        if current_user_id():
            u = db.execute(select(User).where(User.id == current_user_id())).scalar_one_or_none()
            from_email_default = u.email if u else ""
        else: from_email_default = ""
    with st.form("contact_form"):
        from_email = st.text_input("Your email", value=from_email_default)
        subject = st.text_input("Subject"); message = st.text_area("Message", height=160)
        ok = st.form_submit_button("Send message", type="primary")
    if ok:
        if not from_email.strip() or "@" not in from_email: st.error("Please enter a valid email so we can reply.")
        elif not subject.strip() or not message.strip(): st.error("Please add a subject and a message.")
        else:
            body = f"From: {from_email}\n\n{message}"
            sent = send_email(SUPPORT_TO, f"[Doggy Diary] {subject.strip()}", body)
            if sent: st.success("Message sent. Thanks for reaching out!")
            else:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                inbox = DATA_DIR / "support_inbox.csv"
                exists = inbox.exists()
                # Append a properly quoted CSV row
                with open(inbox, "a", encoding="utf-8", newline="") as f:
                    import csv
                    writer = csv.writer(f)
                    if not exists:
                        writer.writerow(["timestamp", "from_email", "subject", "message"])
                        writer.writerow([ts, from_email, subject, message])
                        st.info("Message saved for review. (Email delivery is not configured on this server.)")
    if st.button("Back"): st.session_state.page="Home"; st.rerun()

# ---------------- Admin ----------------
def delete_user_everything(db: Session, user_id: str):
    # Remove dependent rows then user
    db.execute(text("DELETE FROM bookings WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM dogs WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM settings WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM capacities WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM daily_caps WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM owner_profile WHERE owner_id=:oid"), {"oid": user_id})
    db.execute(text("DELETE FROM password_resets WHERE user_id=:uid"), {"uid": user_id})
    db.execute(text("DELETE FROM users WHERE id=:uid"), {"uid": user_id})

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
        c[0].metric("Users", f"{total_users}"); c[1].metric("Dogs", f"{total_dogs}"); c[2].metric("Bookings", f"{total_bookings}")
        st.markdown("### Users")
        q = st.text_input("Search users by email", value="")
        users = db.execute(select(User).order_by(User.created_at.desc())).scalars().all()
        if q.strip(): users = [u for u in users if q.lower() in (u.email or "").lower()]
        for u in users:
            status, days_left, exp_at = sub_status_tuple(u)
            with st.container():
                cols = st.columns([0.22, 0.17, 0.18, 0.16, 0.12, 0.08, 0.07])
                cols[0].write(f"**{u.email}**")
                cols[1].write(f"Created: {u.created_at:%Y-%m-%d}" if u.created_at else "Created: —")
                # Subscription editor
                dval = exp_at.date() if exp_at else (datetime.now().date())
                new_date = cols[2].date_input("Expiry", value=dval, key=f"exp_{u.id}")
                comped = cols[3].checkbox("Lifetime", value=bool(u.sub_comped), key=f"comp_{u.id}")
                if cols[4].button("Save", key=f"save_{u.id}"):
                    u.sub_comped = bool(comped)
                    if not u.sub_comped:
                        u.sub_expires_at = datetime.combine(new_date, time(23,59), tzinfo=timezone.utc)
                    db.commit(); st.success("Saved ✅"); st.rerun()
                if cols[5].button("Impersonate", key=f"imp_{u.id}"):
                    st.session_state["user_id"] = u.id
                    st.session_state["user_name"] = u.full_name or u.email
                    st.session_state["impersonated_by_admin"] = True
                    st.session_state.page = "Home"; st.rerun()
                if cols[6].button("🗑️", key=f"del_{u.id}"):
                    delete_user_everything(db, u.id); db.commit()
                    st.warning(f"Deleted {u.email} and all data."); st.rerun()

        st.markdown("---")
        st.subheader("Global Bookings (read-only)")
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
        if gdf.empty: st.info("No bookings in the global window.")
        else: st.dataframe(gdf, use_container_width=True, height=280)

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

def subscribe_page():
    if not current_user_id() and not is_admin():
        auth_page(); return
    topbar()
    st.header("🔒 Your free trial has ended")
    with SessionLocal() as db:
        if current_user_id():
            u = db.get(User, current_user_id())
            ensure_trial_if_missing(db, u)
            status, days_left, exp_at = sub_status_tuple(u)
            if status in ("trial","active","comped"):
                st.success("You already have access. Enjoy!")
                st.session_state.page = "Home"; st.rerun()
            st.write("You're on the free plan with **no card required** during your trial.")
            if exp_at:
                st.write(f"Your access expired on **{exp_at.date().isoformat()}**.")
    st.markdown("To continue using Doggy Diary, please subscribe for **£12/year**.")
    if SUBSCRIBE_URL:
        st.link_button("Subscribe £12/year", SUBSCRIBE_URL, type="primary", use_container_width=True)
    else:
        st.warning("Admin note: set SUBSCRIBE_URL in secrets or environment to enable one-click checkout.")
    st.caption("Once subscribed, you’ll regain full access immediately.")

def main():
    if "page" not in st.session_state: st.session_state.page = "Home"
    sidebar_nav()
    p = st.session_state.page
    if p == "Admin": admin_page()
    elif p == "Home":
        if not current_user_id() and not is_admin(): auth_page()
        else:
            if current_user_id(): entitlement_guard()
            nav_home()
    elif p == "Recover Account": recover_account_page()
    elif p == "Reset Password": reset_password_page()
    elif p == "Contact Us": contact_page()
    elif p == "Subscribe": subscribe_page()
    elif p == "My Profile":
        if not current_user_id(): auth_page()
        else: my_profile_section()
    elif p == "Doggy Profiles": dogs_section()
    elif p == "Bookings": bookings_section()
    elif p == "Calendar": calendar_section()
    elif p == "Settings": settings_section()
    elif p == "Export": export_section()
    else: nav_home()

if __name__ == "__main__":
    main()
