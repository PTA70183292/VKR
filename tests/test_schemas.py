import pytest
from pydantic import ValidationError
from schemas import PredictRequest, PredictResponse
from datetime import datetime

class TestPredictRequest:

    
    def test_valid_request(self):

        request = PredictRequest(
            user_id="test_user",
            text="Test text"
        )
        
        assert request.user_id == "test_user"
        assert request.text == "Test text"
    
    def test_missing_user_id(self):

        with pytest.raises(ValidationError):
            PredictRequest(text="Test text")
    
    def test_missing_text(self):

        with pytest.raises(ValidationError):
            PredictRequest(user_id="test_user")
    
    def test_empty_strings(self):

        request = PredictRequest(
            user_id="",
            text=""
        )
        
        assert request.user_id == ""
        assert request.text == ""
    
    def test_unicode_text(self):

        request = PredictRequest(
            user_id="test_user",
            text="Привет мир!"
        )
        
        assert "Привет" in request.text
        assert "Привет" in request.text
        assert "🎉" in request.text

class TestPredictResponse:

    
    def test_valid_response(self):

        response = PredictResponse(
            id=1,
            user_id="test_user",
            text="Test text",
            label="LABEL_0",
            score=0.95,
            created_at=datetime.now()
        )
        
        assert response.id == 1
        assert response.user_id == "test_user"
        assert response.text == "Test text"
        assert response.label == "LABEL_0"
        assert response.score == 0.95
        assert isinstance(response.created_at, datetime)
    
    def test_score_validation(self):
        response = PredictResponse(
            id=1,
            user_id="test_user",
            text="Test",
            label="LABEL_0",
            score=1.5, 
            created_at=datetime.now()
        )
        
        assert response.score == 1.5
    
    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            PredictResponse(
                id=1,
                user_id="test_user"
            )
    
    def test_wrong_types(self):
        with pytest.raises(ValidationError):
            PredictResponse(
                id="not_an_int",
                user_id="test_user",
                text="Test",
                label="LABEL_0",
                score=0.95,
                created_at=datetime.now()
            )
    
    def test_from_orm(self):
        from models import PredictionRecord
        

        class MockPrediction:
            id = 1
            user_id = "test_user"
            text = "Test text"
            label = "LABEL_0"
            score = 0.95
            created_at = datetime.now()
        
        mock_obj = MockPrediction()
        
        response = PredictResponse.model_validate(mock_obj)
        
        assert response.id == mock_obj.id
        assert response.user_id == mock_obj.user_id
