# CardioRisk AI

## Run locally
```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open the application at `http://127.0.0.1:8000`.

## GitHub
```bash
git init
git add .
git commit -m "Initial CardioRisk AI deployment"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

See `DEPLOY.md` for Vercel setup.
