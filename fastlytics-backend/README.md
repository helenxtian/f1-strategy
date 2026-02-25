# fastlytics-backend

Real-data FastAPI backend for Fastlytics using Fast-F1.

## 1) Create environment

```bash
cd fastlytics-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Configure env

```bash
cp .env.example .env
```

## 3) Run server

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

API root: http://localhost:8000/

## Notes

- Fast-F1 will download session data on first request, so initial calls may be slower.
- Frontend should point `VITE_API_BASE_URL` to `http://localhost:8000`.
- If your frontend runs on `5173`, CORS is already included.
