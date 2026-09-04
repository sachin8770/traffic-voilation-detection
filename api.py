"""
api.py

Minimal FastAPI service to receive finalized plate reads from the ANPR
pipeline and store them in a database.

Run:
    pip install fastapi uvicorn sqlalchemy
    uvicorn api:app --reload

Swap SQLITE_URL for a Postgres URL when you're ready for production, e.g.:
    postgresql+psycopg2://user:password@localhost/anpr_db
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./plates.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PlateRecord(Base):
    __tablename__ = "plate_records"

    id = Column(Integer, primary_key=True, index=True)
    car_id = Column(Integer, index=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)
    num_readings = Column(Integer)
    video_source = Column(String, nullable=True)
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)

app = FastAPI(title="ANPR Plate Ingest API")

import subprocess

@app.get("/")
def read_root():
    return {"status": "online", "message": "Traffic violation backend is running. Use /start-video to trigger video processing."}

@app.get("/start-video")
def start_video():
    # Launch main.py in the background
    subprocess.Popen(["python", "main.py"])
    return {"status": "started", "message": "Video processing started in the background. It will send violations to your Next.js app!"}


class PlateIn(BaseModel):
    car_id: int
    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    num_readings: int = 1
    video_source: str | None = None


class PlateOut(PlateIn):
    id: int
    detected_at: datetime

    class Config:
        from_attributes = True


@app.post("/plates", response_model=PlateOut)
def create_plate(plate: PlateIn):
    db = SessionLocal()
    try:
        record = PlateRecord(**plate.model_dump())
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


@app.get("/plates", response_model=list[PlateOut])
def list_plates(car_id: int | None = None, limit: int = 100):
    db = SessionLocal()
    try:
        q = db.query(PlateRecord)
        if car_id is not None:
            q = q.filter(PlateRecord.car_id == car_id)
        return q.order_by(PlateRecord.detected_at.desc()).limit(limit).all()
    finally:
        db.close()


@app.get("/plates/{plate_id}", response_model=PlateOut)
def get_plate(plate_id: int):
    db = SessionLocal()
    try:
        record = db.query(PlateRecord).filter(PlateRecord.id == plate_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Not found")
        return record
    finally:
        db.close()
