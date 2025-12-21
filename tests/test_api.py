import pytest
from fastapi import status

class TestHealthCheck:
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "ok"}

class TestPredictEndpoint:
    
    def test_predict_success(self, client, sample_texts):
        payload = {
            "user_id": "test_user_1",
            "text": sample_texts["positive"]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "id" in data
        assert "user_id" in data
        assert "text" in data
        assert "label" in data
        assert "score" in data
        assert "created_at" in data
        
        assert isinstance(data["id"], int)
        assert isinstance(data["user_id"], str)
        assert isinstance(data["text"], str)
        assert isinstance(data["label"], str)
        assert isinstance(data["score"], float)
        
        assert data["user_id"] == "test_user_1"
        assert data["text"] == sample_texts["positive"]
        assert 0.0 <= data["score"] <= 1.0
    
    def test_predict_empty_text(self, client):
        payload = {
            "user_id": "test_user_2",
            "text": ""
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_200_OK
    
    def test_predict_missing_user_id(self, client, sample_texts):
        payload = {
            "text": sample_texts["neutral"]
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_missing_text(self, client):
        payload = {
            "user_id": "test_user_3"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_long_text(self, client):
        payload = {
            "user_id": "test_user_4",
            "text": "Отличный продукт! " * 100
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_200_OK

class TestGetPredictions:
    
    def test_get_user_predictions_empty(self, client):
        response = client.get("/predictions/user/nonexistent_user")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []
    
    def test_get_user_predictions_with_data(self, client, sample_texts):
        user_id = "test_user_5"
        
        for text in sample_texts.values():
            client.post("/predict", json={"user_id": user_id, "text": text})
        
        response = client.get(f"/predictions/user/{user_id}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert len(data) == 3
        assert all(pred["user_id"] == user_id for pred in data)
    
    def test_get_user_predictions_with_pagination(self, client, sample_texts):
        user_id = "test_user_6"
        
        for i in range(5):
            client.post("/predict", json={
                "user_id": user_id,
                "text": f"Test text {i}"
            })
        
        response = client.get(f"/predictions/user/{user_id}?limit=3")
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 3
        
        response = client.get(f"/predictions/user/{user_id}?skip=2&limit=3")
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 3
    
    def test_get_prediction_by_id(self, client, sample_texts):
        payload = {
            "user_id": "test_user_7",
            "text": sample_texts["positive"]
        }
        create_response = client.post("/predict", json=payload)
        prediction_id = create_response.json()["id"]
        
        response = client.get(f"/predictions/{prediction_id}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["id"] == prediction_id
        assert data["user_id"] == "test_user_7"
    
    def test_get_prediction_not_found(self, client):
        response = client.get("/predictions/99999")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_all_predictions(self, client, sample_texts):

        for i, text in enumerate(sample_texts.values()):
            client.post("/predict", json={
                "user_id": f"user_{i}",
                "text": text
            })
        
        response = client.get("/predictions")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert len(data) >= 3

class TestMultipleUsers:
    
    def test_different_users_isolated(self, client, sample_texts):
        user1_id = "user_isolated_1"
        user2_id = "user_isolated_2"
        
        client.post("/predict", json={
            "user_id": user1_id,
            "text": sample_texts["positive"]
        })
        
        client.post("/predict", json={
            "user_id": user2_id,
            "text": sample_texts["negative"]
        })
        
        response1 = client.get(f"/predictions/user/{user1_id}")
        response2 = client.get(f"/predictions/user/{user2_id}")
        
        assert len(response1.json()) == 1
        assert len(response2.json()) == 1
        assert response1.json()[0]["user_id"] == user1_id
        assert response2.json()[0]["user_id"] == user2_id
