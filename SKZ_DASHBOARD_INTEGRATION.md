# SKZ Dashboard Integration Guide

## Overview
The SKZ Dashboard is a React-based visualization dashboard that provides real-time monitoring and visualization of the 7 autonomous agents workflow system. It has been integrated into the Open Journal Systems (OJS) interface to provide seamless access to agent analytics and workflow monitoring.

## Features
- **Agent Status Monitoring**: Real-time status and performance metrics for all 7 autonomous agents
- **Workflow Visualization**: Interactive visualization of manuscript processing workflows
- **Network Analysis**: Agent interaction patterns and communication flows
- **Cognitive Architecture**: Hierarchical and distributed agent architecture visualization
- **Analytics Dashboard**: Performance metrics, success rates, and trend analysis

## Integration Architecture

### Components
1. **React Dashboard** (`skz-integration/workflow-visualization-dashboard/`)
   - Modern React 19 application with TypeScript support
   - Built with Vite for optimal performance
   - Uses D3.js for data visualization and TailwindCSS for styling
   - Responsive design with mobile support

2. **OJS Integration** (`pages/skzDashboard/`)
   - PHP handler class for routing and template management
   - Template integration with OJS theme system
   - Navigation menu integration

3. **Static Assets** (`public/skz-dashboard/`)
   - Built React application served as static files
   - Optimized production builds with asset bundling

### File Structure
```
/
├── pages/skzDashboard/
│   └── SkzDashboardHandler.inc.php      # PHP handler for dashboard routes
├── templates/skzDashboard/
│   └── index.tpl                        # OJS template for dashboard
├── public/skz-dashboard/                # Built React app (auto-generated)
├── skz-integration/workflow-visualization-dashboard/
│   ├── src/                             # React source code
│   ├── public/                          # Static assets
│   └── dist/                            # Built application
└── deploy-skz-dashboard.sh              # Deployment script
```

## Deployment

### Automated Deployment
Run the deployment script to build and deploy the dashboard:
```bash
./deploy-skz-dashboard.sh
```

### Manual Deployment
1. Navigate to the dashboard directory:
   ```bash
   cd skz-integration/workflow-visualization-dashboard
   ```

2. Install dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

3. Build the application:
   ```bash
   npm run build
   ```

4. Copy to OJS public directory:
   ```bash
   cp -r dist/* ../../public/skz-dashboard/
   ```

## Accessing the Dashboard

### Navigation
The dashboard is accessible through the main OJS navigation menu:
- **Main Dashboard**: Journal → SKZ Dashboard → Overview
- **Agent Status**: Journal → SKZ Dashboard → Agent Status
- **Workflow Visualization**: Journal → SKZ Dashboard → Workflow Visualization
- **Analytics**: Journal → SKZ Dashboard → Analytics

### Direct URLs
- Main Dashboard: `[ojs-url]/index.php/[journal]/skzDashboard`
- Agent Status: `[ojs-url]/index.php/[journal]/skzDashboard/agents`
- Workflow: `[ojs-url]/index.php/[journal]/skzDashboard/workflow`
- Analytics: `[ojs-url]/index.php/[journal]/skzDashboard/analytics`

## Configuration

### Permissions
The dashboard is accessible to all authenticated users with the following roles:
- Site Administrator
- Journal Manager
- Sub Editor
- Assistant
- Reviewer
- Author

### Customization
To customize the dashboard:
1. Modify React components in `skz-integration/workflow-visualization-dashboard/src/`
2. Update styling in the component files or `src/App.css`
3. Rebuild and redeploy using the deployment script

## Technical Details

### React Integration
The dashboard mounts to a specific container element (`skz-dashboard-root`) within the OJS template system, allowing seamless integration without conflicts.

### Asset Management
- CSS and JavaScript assets are automatically included via OJS template system
- Production builds are optimized and minified
- Assets are served from OJS public directory for consistent loading

### Browser Compatibility
- Modern browsers (Chrome 80+, Firefox 75+, Safari 13+, Edge 80+)
- Responsive design for mobile and tablet devices
- Progressive loading with fallback for JavaScript disabled

## Troubleshooting

### Common Issues
1. **Dashboard not loading**: Ensure assets are correctly copied to `public/skz-dashboard/`
2. **Navigation not showing**: Check journal context and user permissions
3. **Build errors**: Run `npm install --legacy-peer-deps` to resolve dependency conflicts

### Debugging
1. Check browser console for JavaScript errors
2. Verify file paths in network tab
3. Ensure OJS handlers are properly registered

## Development

### Local Development
1. Start the React development server:
   ```bash
   cd skz-integration/workflow-visualization-dashboard
   npm run dev
   ```

2. For OJS integration testing, use the deployment script after making changes

### Adding New Features
1. Develop new components in React application
2. Test in development mode
3. Build and deploy to OJS
4. Update documentation as needed

## Maintenance

### Updates
- Run `./deploy-skz-dashboard.sh` after any changes to the React application
- Monitor browser console for any errors
- Regular testing of all dashboard features

### Performance
- Dashboard assets are optimized for production
- Consider implementing service workers for offline functionality
- Monitor bundle sizes and optimize as needed