"""
BLINK — SQLAlchemy Database Models for PostgreSQL
Mirror schemas representing users (E2EE envelopes) and attendance records.
"""
from sqlalchemy import Column, String, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class PostgresUser(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    # Stores E2EE envelope: { "ciphertext": "...", "iv": "...", "tag": "..." }
    # Only the mobile devices have the decryption key!
    encrypted_envelope_json = Column(JSON, nullable=True) 
    model_mode = Column(String, nullable=False, default='simulated')
    enrolled_at = Column(String, nullable=False)
    status = Column(String, nullable=False, default='ACTIVE')

    attendance_records = relationship("PostgresAttendance", back_populates="user", cascade="all, delete-orphan")

class PostgresAttendance(Base):
    __tablename__ = 'attendance'
    
    id = Column(String, primary_key=True)
    person_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    timestamp = Column(String, nullable=False)
    status = Column(String, nullable=False, default='CHECK_IN')
    punctuality = Column(String, nullable=False, default='ON_TIME')

    user = relationship("PostgresUser", back_populates="attendance_records")
