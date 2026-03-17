# Deployment Guide for Vercel

## Architecture Overview

┌─────────────────────────────────────────────────────────────────────┐
│                         VERCEL DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │   Frontend       │         │   Backend        │                 │
│  │   (Static)       │         │   (Serverless)   │                 │
│  │                  │         │                  │                 │
│  │  • React + Vite  │         │  • FastAPI       │                 │
│  │  • Maps/Charts   │         │  • ML Models     │                 │
│  │  • UI Components │◄───────►│  • API Routes    │                 │
│  └────────┬─────────┘         └────────┬─────────┘                 │
│           │                          │                             │
│           │    ┌──────────────────┐  │                             │
│           └───►│  Vercel Edge     │  │                             │
│                │  Network         │◄─┘                             │
│                └──────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

## Quick Start

1. Install Vercel CLI: `npm i -g vercel`
2. Login: `vercel login`
3. Deploy: `vercel --prod`

## Environment Variables

Set these in Vercel Dashboard → Project Settings → Environment Variables:

```
DATABASE_URL=<optional-postgres-url>
OPENAQ_API_KEY=<your-openaq-api-key>
ENVIRONMENT=production
```

## Pre-Deployment Checklist

- [ ] All ML models in /models directory
- [ ] data/output/recommendations.json exists
- [ ] data/raw/factories/factories.csv exists
- [ ] data/raw/pollution/pollution_readings.csv exists
- [ ] Environment variables configured
- [ ] Build commands tested locally

## File Structure for Deployment

```
smart-factory-emission-monitoring/
├── vercel.json              # Vercel configuration
├── requirements-vercel.txt # Production dependencies
├── frontend-vite/
│   ├── vercel.json         # Frontend config
│   └── .env.production     # Production env vars
├── backend/
│   └── vercel.json         # Backend config
└── models/                 # ML models (included in deployment)
    ├── pollution_impact_model.pkl
    ├── pollution_forecast_model.pkl
    └── recommendation_model.pkl
```

## Important Notes

1. **ML Models**: Must be < 50MB each for serverless functions
2. **Data Files**: Large CSVs should be hosted externally (S3/Cloudflare R2)
3. **Cold Starts**: Serverless functions have cold starts (1-3 seconds)
4. **Database**: Use Vercel Postgres or external service (Supabase, Neon)

## Alternative: Separate Deployments

If Vercel Serverless has limitations:

**Option A: Frontend on Vercel + Backend on Render/Railway**
- Frontend: https://smart-factory-emission-monitoring.vercel.app
- Backend: https://smart-factory-api.onrender.com

**Option B: Docker Container on Railway/Render**
- Deploy full stack as single container
- Better for large ML models

## Support

For issues:
1. Check Vercel Function Logs (Dashboard → Functions)
2. Verify environment variables are set
3. Check ML model file sizes (< 50MB)
4. Test API endpoints: `/api/health`
