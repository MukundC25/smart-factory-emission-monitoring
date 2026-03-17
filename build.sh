#!/bin/bash
# Build script for Vercel deployment

echo "🚀 Starting Vercel build process..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements-vercel.txt

# Verify ML models exist
echo "🔍 Checking ML models..."
if [ ! -f "models/pollution_impact_model.pkl" ]; then
    echo "❌ Error: pollution_impact_model.pkl not found"
    exit 1
fi

if [ ! -f "models/pollution_forecast_model.pkl" ]; then
    echo "❌ Error: pollution_forecast_model.pkl not found"
    exit 1
fi

if [ ! -f "models/recommendation_model.pkl" ]; then
    echo "❌ Error: recommendation_model.pkl not found"
    exit 1
fi

# Check data files
echo "📂 Checking data files..."
if [ ! -f "data/raw/factories/factories.csv" ]; then
    echo "⚠️  Warning: factories.csv not found"
fi

if [ ! -f "data/output/recommendations.json" ]; then
    echo "⚠️  Warning: recommendations.json not found - run: python -m src.recommendations.generate_all"
fi

echo "✅ Build check complete!"
echo ""
echo "📊 Deployment Summary:"
echo "  • ML Models: Ready"
echo "  • Backend: FastAPI"
echo "  • Frontend: Vite + React"
echo ""
echo "🌐 After deployment, verify at:"
echo "  Health Check: https://your-project.vercel.app/health"
echo "  API Docs: https://your-project.vercel.app/docs"
