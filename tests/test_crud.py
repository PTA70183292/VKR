import pytest
import crud
from models import PredictionRecord

class TestCreatePrediction:
    
    def test_create_prediction_success(self, test_db):
        prediction = crud.create_prediction(
            db=test_db,
            user_id="test_user",
            text="Test text",
            label="LABEL_0",
            score=0.95
        )
        
        assert prediction.id is not None
        assert prediction.user_id == "test_user"
        assert prediction.text == "Test text"
        assert prediction.label == "LABEL_0"
        assert prediction.score == 0.95
        assert prediction.created_at is not None
    
    def test_create_multiple_predictions(self, test_db):
        for i in range(3):
            prediction = crud.create_prediction(
                db=test_db,
                user_id=f"user_{i}",
                text=f"Text {i}",
                label=f"LABEL_{i}",
                score=0.8 + i * 0.05
            )
            assert prediction.id == i + 1

class TestGetPredictions:
    
    def test_get_predictions_by_user(self, test_db):
        user_id = "test_user_1"
        
        for i in range(3):
            crud.create_prediction(
                db=test_db,
                user_id=user_id,
                text=f"Text {i}",
                label="LABEL_0",
                score=0.9
            )
        
        predictions = crud.get_predictions_by_user(test_db, user_id)
        assert len(predictions) == 3
        assert all(p.user_id == user_id for p in predictions)
    
    def test_get_predictions_with_skip_limit(self, test_db):
        user_id = "test_user_2"
        
        for i in range(10):
            crud.create_prediction(
                db=test_db,
                user_id=user_id,
                text=f"Text {i}",
                label="LABEL_0",
                score=0.9
            )
        
        predictions = crud.get_predictions_by_user(test_db, user_id, skip=5, limit=3)
        assert len(predictions) == 3
    
    def test_get_prediction_by_id(self, test_db):
        prediction = crud.create_prediction(
            db=test_db,
            user_id="test_user",
            text="Test",
            label="LABEL_0",
            score=0.9
        )
        
        found = crud.get_prediction_by_id(test_db, prediction.id)
        assert found is not None
        assert found.id == prediction.id
        assert found.text == "Test"
    
    def test_get_prediction_by_id_not_found(self, test_db):
        found = crud.get_prediction_by_id(test_db, 99999)
        assert found is None
    
    def test_get_all_predictions(self, test_db):
        for i in range(5):
            crud.create_prediction(
                db=test_db,
                user_id=f"user_{i}",
                text=f"Text {i}",
                label="LABEL_0",
                score=0.9
            )
        
        predictions = crud.get_all_predictions(test_db)
        assert len(predictions) == 5
    
    def test_get_all_predictions_with_limit(self, test_db):
        for i in range(10):
            crud.create_prediction(
                db=test_db,
                user_id=f"user_{i}",
                text=f"Text {i}",
                label="LABEL_0",
                score=0.9
            )
        
        predictions = crud.get_all_predictions(test_db, limit=5)
        assert len(predictions) == 5

class TestPredictionModel:
    
    def test_prediction_repr(self, test_db):
        prediction = crud.create_prediction(
            db=test_db,
            user_id="test_user",
            text="Test",
            label="LABEL_0",
            score=0.95
        )
        
        repr_str = repr(prediction)
        assert "test_user" in repr_str
        assert "LABEL_0" in repr_str
        assert "0.95" in repr_str
