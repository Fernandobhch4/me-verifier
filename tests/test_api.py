from api.app import app

def test_healthz():
    c = app.test_client()
    r = c.get("/healthz")
    assert r.status_code == 200

def test_verify_no_file():
    c = app.test_client()
    r = c.post("/verify", data={})
    assert r.status_code == 400
