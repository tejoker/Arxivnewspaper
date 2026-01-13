import sqlite3
from sqlalchemy import create_engine, Column, String, Integer, Text, Date, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'

    id = Column(String, primary_key=True)  # Arxiv ID
    title = Column(String)
    abstract = Column(Text)
    authors = Column(String)  # JSON or comma-separated
    link = Column(String)
    published_date = Column(Date)
    updated_date = Column(Date)
    categories = Column(String)
    
    # Store embedding as binary blob
    embedding = Column(LargeBinary, nullable=True)
    
    # Store benchmarks as JSON string or comma-separated
    benchmarks = Column(String, nullable=True)

    def __repr__(self):
        return f"<Paper(title='{self.title}', date='{self.published_date}')>"

class Citation(Base):
    __tablename__ = 'citations'
    
    # We use a simple adjacency list
    citing_id = Column(String, primary_key=True) 
    cited_id = Column(String, primary_key=True)
    
    # Optional: Source of this link (e.g. 'openalex', 'matt_bierbaum_2019')
    source = Column(String, nullable=True)

# Setup
DB_PATH = 'sqlite:///journarixv.db'

def get_engine():
    return create_engine(DB_PATH)

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
