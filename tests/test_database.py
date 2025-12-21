import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from database import Base, get_db, init_db
from models import PredictionRecord

class TestDatabase:
    
    def test_get_db_generator(self):
        db_gen = get_db()
        db = next(db_gen)
        
        assert db is not None
        
        try:
            next(db_gen)
        except StopIteration:
            pass
    
    def test_init_db_creates_tables(self):
        test_engine = create_engine("sqlite:///./test_init.db")
        
        Base.metadata.bind = test_engine
        Base.metadata.create_all(bind=test_engine)
        
        inspector = inspect(test_engine)
        tables = inspector.get_table_names()
        
        assert "predictions" in tables
        
        Base.metadata.drop_all(bind=test_engine)
    
    def test_prediction_table_columns(self):
        test_engine = create_engine("sqlite:///./test_columns.db")
        Base.metadata.create_all(bind=test_engine)
        
        inspector = inspect(test_engine)
        columns = [col['name'] for col in inspector.get_columns('predictions')]
        
        expected_columns = ['id', 'user_id', 'text', 'label', 'score', 'created_at']
        for col in expected_columns:
            assert col in columns
        
        Base.metadata.drop_all(bind=test_engine)

class TestPredictionModel:
    
    def test_create_prediction_record(self, test_db):
        prediction = PredictionRecord(
            user_id="test_user",
            text="Test text",
            label="LABEL_0",
            score=0.95
        )
        
        test_db.add(prediction)
        test_db.commit()
        test_db.refresh(prediction)
        
        assert prediction.id is not None
        assert prediction.created_at is not None
    
    def test_prediction_nullable_constraints(self, test_db):
        from sqlalchemy.exc import IntegrityError
        
        prediction = PredictionRecord(
            text="Test",
            label="LABEL_0",
            score=0.9
        )
        
        test_db.add(prediction)
        
        with pytest.raises(IntegrityError):
            test_db.commit()
        
        test_db.rollback()
    
    def test_prediction_score_type(self, test_db):
        prediction = PredictionRecord(
            user_id="test_user",
            text="Test",
            label="LABEL_0",
            score=0.999999
        )
        
        test_db.add(prediction)
        test_db.commit()
        test_db.refresh(prediction)
        
        assert isinstance(prediction.score, float)
        assert 0.0 <= prediction.score <= 1.0
