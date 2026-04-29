# backend/models.py
import enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.sql import func
from database import Base

class InvoiceStatus(str, enum.Enum):
    PENDING   = "PENDING"
    CONFIRMED = "CONFIRMED"

class Invoice(Base):
    __tablename__ = "invoices"

    id                = Column(Integer, primary_key=True, index=True)
    invoice_id        = Column(String, unique=True, index=True)
    original_filename = Column(String, nullable=True)
    s3_key            = Column(String, nullable=True)
    company           = Column(String, nullable=True)
    date              = Column(String, nullable=True)
    total             = Column(String, nullable=True)
    address           = Column(String, nullable=True)
    company_score     = Column(Float, nullable=True)
    date_score        = Column(Float, nullable=True)
    total_score       = Column(Float, nullable=True)
    address_score     = Column(Float, nullable=True)
    processing_time   = Column(Float, nullable=True)
    status            = Column(Enum(InvoiceStatus), default=InvoiceStatus.PENDING)
    created_at        = Column(DateTime(timezone=True), server_default=func.now())
    updated_at        = Column(DateTime(timezone=True), onupdate=func.now())