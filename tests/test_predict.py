VALID_PAYLOAD = {
    "alcohol": 13.5, "malic_acid": 1.8, "ash": 2.4,
    "alcalinity_of_ash": 18.0, "magnesium": 105.0,
    "total_phenols": 2.7, "flavanoids": 2.9,
    "nonflavanoid_phenols": 0.28, "proanthocyanins": 1.85,
    "color_intensity": 5.5, "hue": 1.05,
    "od280/od315_of_diluted_wines": 3.2, "proline": 1180.0,
}

def test_predict_happy_path(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in ("class_0", "class_1", "class_2")
    assert set(body['probabilities'].keys()) == {"class_0", "class_1", "class_2"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.001

def test_predict_negative_alcohol_return_422(client):
    payload = {**VALID_PAYLOAD, "alcohol": -5.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    body = response.json()
    assert any("alcohol" in str(err).lower() for err in body["detail"])

def test_gradio_root_renders(client):
    # GET / отдает HTML с Gradio-приложением
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Gradio в шаблоне содержит свой маркер
    assert "gradio" in response.text.lower()