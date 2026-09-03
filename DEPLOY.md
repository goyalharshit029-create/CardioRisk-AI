# CardioRisk AI — Vercel deployment

## What is already fixed
- Root-level `main.py` exposes the FastAPI `app` for Vercel.
- Frontend is served by the same FastAPI deployment.
- `frontend/app.js` uses the same origin instead of a placeholder Render URL.
- Required runtime `.pkl` ML models are no longer ignored by Git.
- Authentication tokens are stateless and work across serverless instances.
- `GEMINI_API_KEY` is not included in Git.

## Before deploying
In Vercel → Project → Settings → Environment Variables, add:

- `GEMINI_API_KEY` (optional; needed for Gemini-powered assistant)
- `GEMINI_MODEL` (optional)
- `APP_SECRET_KEY` (required for production; use a long random value)

## Important database note
SQLite is suitable for local development. On Vercel, the filesystem is ephemeral, so accounts and assessment history should not be treated as permanently stored. For a production deployment, connect a managed database and set `DATABASE_PATH`/database integration accordingly.

## GitHub
Commit the runtime files:
- `backend/cardiovascular_model.pkl`
- `backend/features.pkl`
- `research/models/clinical_xgboost.pkl`

Do not commit:
- `backend/.env`
- `backend/healthcare.db`
