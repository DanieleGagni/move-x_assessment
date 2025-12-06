"""
SQLAlchemy ORM models for the Bike Sharing database.
"""
from sqlalchemy import Column, Integer, Float, Date, String
from app.database import Base


class BikeRental(Base):
    """
    ORM model representing hourly bike rental data.
    
    Maps to the 'bike_rentals' table in the database.
    Based on the UCI Bike Sharing Dataset structure.
    """
    __tablename__ = "bike_rentals"
    
    # Primary key - record index
    instant = Column(Integer, primary_key=True, index=True)
    
    # Date and time
    dteday = Column(Date, nullable=False, index=True)  # Date
    hr = Column(Integer, nullable=False)  # Hour (0-23)
    
    # Season and year
    season = Column(Integer, nullable=False)  # 1:spring, 2:summer, 3:fall, 4:winter
    yr = Column(Integer, nullable=False)  # Year (0: 2011, 1: 2012)
    mnth = Column(Integer, nullable=False)  # Month (1-12)
    
    # Day information
    holiday = Column(Integer, nullable=False)  # Is holiday (0/1)
    weekday = Column(Integer, nullable=False)  # Day of week (0-6)
    workingday = Column(Integer, nullable=False)  # Is working day (0/1)
    
    # Weather conditions
    weathersit = Column(Integer, nullable=False)  # Weather situation (1-4)
    temp = Column(Float, nullable=False)  # Normalized temperature
    atemp = Column(Float, nullable=False)  # Normalized feeling temperature
    hum = Column(Float, nullable=False)  # Normalized humidity
    windspeed = Column(Float, nullable=False)  # Normalized wind speed
    
    # Rental counts
    casual = Column(Integer, default=0)  # Count of casual users
    registered = Column(Integer, default=0)  # Count of registered users
    cnt = Column(Integer, nullable=False)  # Total rental count
    
    def __repr__(self):
        return f"<BikeRental(date={self.dteday}, hour={self.hr}, count={self.cnt})>"
