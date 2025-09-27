from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.types import JSON, Integer, String
from sqlalchemy import Column, ForeignKey, Table

DEFAULT_DB_URL = "sqlite:///wikindex.db"
TRANSACTION_BATCH_SIZE = 10000

Base = declarative_base()

document_links = Table(
    "document_links",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("from_title", ForeignKey("documents.title"), nullable=False, index=True),
    Column("to_title", ForeignKey("documents.title"), nullable=False, index=True),
)
    
class FileRecord(Base):
    __tablename__ = 'files'
    path = Column(String, primary_key=True)

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True, index=True, nullable=False)
    url = Column(String)
    
    links = relationship(
        "Document",
        secondary=document_links,
        primaryjoin=title == document_links.c.from_title,
        secondaryjoin=title == document_links.c.to_title,
        backref="linked_from"
    )
    # foreign file
    file_path = Column(String, ForeignKey('files.path'), index=True)
    file = relationship("FileRecord", backref="documents")
    
class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey('documents.id'), index=True)
    content = Column(String)

    document = relationship("Document", backref="chunks")


class Client:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._active_session = None
        
    def open(self):
        if self._active_session:
            raise RuntimeError("Session already active.")
        Base.metadata.create_all(self.engine)
        self._active_session = self.SessionLocal()
        return self
    
    def close(self):
        if self._active_session:
            self._active_session.close()
            self._active_session = None
        else:
            raise RuntimeError("No active session to close.")

    def __enter__(self):
        return self.open()
    
    def __exit__(self, *_):
        self.close()
        
    @property
    def session(self):
        if not self._active_session:
            raise RuntimeError("No active session. Use 'with Client(...) as client:' or call open() first.")
        return self._active_session