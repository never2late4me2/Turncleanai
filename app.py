import os
import math
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Literal

from fastapi import (
    FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException, Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Index
)
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from pydantic import BaseModel, Field, field_validator

import stripe

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_TITLE = "TurnAI Clean - Production Backend"
APP_DESC = "AI-enhanced marketplace for short-term rental turnover cleaning"
APP_VERSION = "1.2.0"

SECRET_KEY = os.getenv("JWT_SECRET", "dev_secret_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./turnai_full_utils.db")
ENGINE_KWARGS = (
    {"connect_args": {"check_same_thread": False}}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(DATABASE_URL, echo=False, **ENGINE_KWARGS)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("turnai")

# -----------------------------------------------------------------------------
# Auth setup
# -----------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_identity(token: str = Depends(oauth2_scheme)) -> Dict[str, str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        role = payload.get("role")
        if not sub or not role:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return {"sub": sub, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -----------------------------------------------------------------------------
# SQLAlchemy models
# -----------------------------------------------------------------------------
class Cleaner(Base):
    __tablename__ = "cleaners"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True, unique=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    rating = Column(Float, default=4.8, nullable=False)
    available = Column(Boolean, default=True, nullable=False)
    stripe_account_id = Column(String(100), unique=True, nullable=True)
    fcm_token = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)

    assigned_jobs = relationship("Job", back_populates="assigned_cleaner")

    __table_args__ = (
        Index("ix_cleaners_location", "lat", "lon"),
        Index("ix_cleaners_available", "available"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    host_id = Column(String(100), nullable=False, index=True)
    description = Column(String(500), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    estimated_price = Column(Float, nullable=False)
    status = Column(String(20), default="pending", nullable=False, index=True)
    assigned_cleaner_id = Column(Integer, ForeignKey("cleaners.id"), nullable=True)
    payment_intent_id = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    assigned_cleaner = relationship("Cleaner", back_populates="assigned_jobs")

    __table_args__ = (
        Index("ix_jobs_location", "lat", "lon"),
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_created", "created_at"),
    )


Base.metadata.create_all(bind=engine)

# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
class Role(str):
    pass  # used for clarity

UserRole = Literal["host", "cleaner"]

class CleanerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    lat: float
    lon: float
    password: str = Field(..., min_length=8)
    stripe_account_id: Optional[str] = None
    fcm_token: Optional[str] = None

    @field_validator("lat")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("lon")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v


class JobCreate(BaseModel):
    description: str = Field(..., min_length=10, max_length=500)
    lat: float
    lon: float

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Description cannot be empty")
        return stripped

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        return v


class JobResponse(BaseModel):
    job_id: int
    estimated_price: float
    status: str
    assigned_to: Optional[int] = None
    issues: List[str] = Field(default_factory=list)
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def analyze_images(files: List[UploadFile]) -> Dict[str, object]:
    """
    Simulated AI vision analysis.
    Replace with Google Cloud Vision API or similar in production.
    """
    base_price = 250.0
    issues = ["simulated clutter in living room", "minor bathroom staining"]
    cleanliness_score = 82  # 0-100
    adjustment = (100 - cleanliness_score) * 3.5
    estimated_price = max(150.0, min(600.0, base_price + adjustment))
    return {
        "estimated_price": round(estimated_price, 2),
        "cleanliness_score": cleanliness_score,
        "issues": issues,
    }


def get_nearby_cleaners(
    db: Session,
    lat: float,
    lon: float,
    max_distance_km: float = 25.0,
    limit: int = 5,
) -> List[Cleaner]:
    cleaners = db.query(Cleaner).filter(Cleaner.available.is_(True)).all()
    nearby = []
    for cleaner in cleaners:
        dist = haversine_distance(lat, lon, cleaner.lat, cleaner.lon)
        if dist <= max_distance_km:
            nearby.append((cleaner, dist))
    nearby_sorted = sorted(nearby, key=lambda x: (x[1], -x[0].rating, x[0].id))
    return [cleaner for cleaner, _ in nearby_sorted][:limit]


async def notify_cleaners(cleaners: List[Cleaner], job_id: int, message: Optional[str] = None):
    """
    Future-ready stub for notifications (FCM/Twilio).
    """
    default_message = f"New turnover job #{job_id} available in your area!"
    msg = message or default_message
    logger.info(f"[NOTIFICATION] Sending to {len(cleaners)} cleaners: {msg}")
    # TODO: Integrate FCM/Twilio here.


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Completed with status {response.status_code}")
    return response


# -----------------------------------------------------------------------------
# Health & support
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/support")
def support():
    return {"ko_fi": "https://ko-fi.com/bryantolbert"}


# -----------------------------------------------------------------------------
# Auth endpoints (JWT)
# -----------------------------------------------------------------------------
@app.post("/token", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(lambda: SessionLocal()),
):
    # For prototype: use cleaner table for login;
    # in production, you may want separate user table for hosts and cleaners.
    user = db.query(Cleaner).filter(Cleaner.name == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Role selection: clients can pass "role" in scope; default to "cleaner"
    role = "cleaner"
    if "host" in (form_data.scopes or []):
        role = "host"

    token = create_access_token({"sub": user.name, "role": role})
    return TokenResponse(access_token=token, token_type="bearer")


# -----------------------------------------------------------------------------
# Cleaner registration
# -----------------------------------------------------------------------------
@app.post(
    "/cleaners/register",
    responses={200: {"description": "Cleaner registered successfully"}},
)
def register_cleaner(
    cleaner_data: CleanerCreate = Body(
        ...,
        examples={
            "basic": {
                "summary": "Simple cleaner",
                "description": "Register a cleaner with coordinates",
                "value": {
                    "name": "Alex",
                    "lat": 36.15,
                    "lon": -95.99,
                    "password": "StrongPass123!"
                },
            },
            "with_stripe": {
                "summary": "Cleaner with Stripe account",
                "value": {
                    "name": "Jordan",
                    "lat": 36.12,
                    "lon": -95.98,
                    "password": "AnotherStrongPwd!9",
                    "stripe_account_id": "acct_12345",
                },
            },
        },
    ),
    db: Session = Depends(lambda: SessionLocal()),
):
    if cleaner_data.stripe_account_id:
        existing = (
            db.query(Cleaner)
            .filter(Cleaner.stripe_account_id == cleaner_data.stripe_account_id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=400, detail="Stripe account already registered")

    existing_name = db.query(Cleaner).filter(Cleaner.name == cleaner_data.name).first()
    if existing_name:
        raise HTTPException(status_code=400, detail="Cleaner name already registered")

    db_cleaner = Cleaner(
        name=cleaner_data.name,
        lat=cleaner_data.lat,
        lon=cleaner_data.lon,
        stripe_account_id=cleaner_data.stripe_account_id,
        fcm_token=cleaner_data.fcm_token,
        password_hash=hash_password(cleaner_data.password),
    )
    db.add(db_cleaner)
    db.commit()
    db.refresh(db_cleaner)
    return {"cleaner_id": db_cleaner.id}


# -----------------------------------------------------------------------------
# Job creation (host-only)
# -----------------------------------------------------------------------------
@app.post(
    "/jobs/create",
    response_model=JobResponse,
)
async def create_job(
    background_tasks: BackgroundTasks,
    job_data: JobCreate = Body(
        ...,
        examples={
            "standard": {
                "summary": "Standard job",
                "description": "2BR turnover job in midtown",
                "value": {"description": "2BR turnover, linens, floors, bathroom deep clean", "lat": 36.15, "lon": -95.99},
            }
        },
    ),
    photos: List[UploadFile] = File(default=[]),
    db: Session = Depends(lambda: SessionLocal()),
    identity: Dict[str, str] = Depends(get_current_identity),
):
    if identity["role"] != "host":
        raise HTTPException(status_code=403, detail="Only hosts can create jobs")

    ai_result = analyze_images(photos)
    new_job = Job(
        host_id=identity["sub"],
        description=job_data.description,
        lat=job_data.lat,
        lon=job_data.lon,
        estimated_price=ai_result["estimated_price"],
        status="pending",
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    nearby = get_nearby_cleaners(db, job_data.lat, job_data.lon)
    assigned_id = None
    if nearby:
        top_cleaner = nearby[0]
        new_job.assigned_cleaner_id = top_cleaner.id
        new_job.status = "assigned"
        db.commit()
        background_tasks.add_task(notify_cleaners, nearby[:3], new_job.id)
        assigned_id = top_cleaner.id

    return JobResponse(
        job_id=new_job.id,
        estimated_price=ai_result["estimated_price"],
        status=new_job.status,
        assigned_to=assigned_id,
        issues=ai_result["issues"],
        created_at=new_job.created_at,
    )


# -----------------------------------------------------------------------------
# Payments: Stripe PaymentIntent for a job
# -----------------------------------------------------------------------------
@app.post("/jobs/{job_id}/payment_intent")
def create_payment_intent(job_id: int, db: Session = Depends(lambda: SessionLocal()), identity: Dict[str, str] = Depends(get_current_identity)):
    # Hosts initiate payment intent
    if identity["role"] != "host":
        raise HTTPException(status_code=403, detail="Only hosts can initiate payments")
    job = db.query(Job).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not stripe.api_key:
        raise HTTPException(status_code=400, detail="Stripe is not configured")

    amount_cents = max(0, int(job.estimated_price * 100))
    intent = stripe.PaymentIntent.create(
        amount=amount_cents,
        currency="usd",
        description=f"TurnAI Clean job #{job.id}",
        metadata={"job_id": job.id, "host_id": job.host_id},
    )
    job.payment_intent_id = intent["id"]
    db.commit()
    return {"payment_intent_id": intent["id"], "client_secret": intent["client_secret"]}


# -----------------------------------------------------------------------------
# Entry points note:
# Run with: uvicorn app:app --reload
# -----------------------------------------------------------------------------
