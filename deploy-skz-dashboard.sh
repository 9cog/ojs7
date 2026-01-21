#!/bin/bash

# SKZ Dashboard Deployment Script
# This script builds the React dashboard and deploys it to OJS public directory

set -e

echo "🚀 Starting SKZ Dashboard deployment..."

# Navigate to dashboard directory
cd "$(dirname "$0")/skz-integration/workflow-visualization-dashboard"

echo "📦 Installing dependencies..."
npm install --legacy-peer-deps --silent

echo "🔧 Building React dashboard..."
npm run build

echo "📁 Deploying to OJS public directory..."
rm -rf ../../public/skz-dashboard
mkdir -p ../../public/skz-dashboard
cp -r dist/* ../../public/skz-dashboard/

echo "🔍 Checking deployment..."
if [ -f "../../public/skz-dashboard/index.html" ]; then
    echo "✅ Dashboard deployed successfully!"
    echo "📊 Dashboard files:"
    ls -la ../../public/skz-dashboard/
else
    echo "❌ Deployment failed!"
    exit 1
fi

echo "🎯 SKZ Dashboard is ready at: /public/skz-dashboard/"
echo "🔗 Access via OJS at: [your-ojs-url]/index.php/[journal]/skzDashboard"