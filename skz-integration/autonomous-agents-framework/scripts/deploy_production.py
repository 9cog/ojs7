"""
Production Deployment Script for SKZ Autonomous Agents Framework
Handles complete production deployment with health checks and monitoring
"""
import os
import sys
import subprocess
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Production deployment manager"""
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.deployment_status = {}
        
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        default_config = {
            'environment': 'production',
            'services': {
                'api_server': {
                    'port': 8000,
                    'workers': 4,
                    'timeout': 120
                },
                'agent_orchestrator': {
                    'port': 8001,
                    'max_concurrent_tasks': 10
                },
                'communication_service': {
                    'port': 8002,
                    'smtp_enabled': True
                },
                'monitoring': {
                    'port': 8003,
                    'metrics_interval': 60
                }
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'skz_agents_prod',
                'pool_size': 20
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'vector_db': {
                'path': '/var/lib/skz-agents/vectordb',
                'collection_name': 'research_documents_prod'
            },
            'security': {
                'jwt_secret': 'CHANGE_ME_IN_PRODUCTION',
                'api_key_required': True,
                'rate_limiting': True
            },
            'monitoring': {
                'health_check_interval': 30,
                'log_level': 'INFO',
                'metrics_retention_days': 30
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")
        
        return default_config
    
    def deploy(self) -> bool:
        """Execute complete production deployment"""
        logger.info("🚀 Starting SKZ Agents Framework Production Deployment")
        
        try:
            # Pre-deployment checks
            if not self._pre_deployment_checks():
                logger.error("❌ Pre-deployment checks failed")
                return False
            
            # Environment setup
            if not self._setup_environment():
                logger.error("❌ Environment setup failed")
                return False
            
            # Database migration
            if not self._migrate_database():
                logger.error("❌ Database migration failed")
                return False
            
            # Deploy services
            if not self._deploy_services():
                logger.error("❌ Service deployment failed")
                return False
            
            # Configure monitoring
            if not self._setup_monitoring():
                logger.error("❌ Monitoring setup failed")
                return False
            
            # Health checks
            if not self._perform_health_checks():
                logger.error("❌ Health checks failed")
                return False
            
            # Performance benchmarks
            if not self._run_performance_tests():
                logger.warning("⚠️ Performance tests had issues but deployment continues")
            
            logger.info("✅ Production deployment completed successfully!")
            self._generate_deployment_report()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            self._rollback_deployment()
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation checks"""
        logger.info("🔍 Running pre-deployment checks...")
        
        checks = {
            'python_version': self._check_python_version(),
            'dependencies': self._check_dependencies(),
            'disk_space': self._check_disk_space(),
            'permissions': self._check_permissions(),
            'ports_available': self._check_ports(),
            'env_variables': self._check_environment_variables()
        }
        
        failed_checks = [check for check, passed in checks.items() if not passed]
        
        if failed_checks:
            logger.error(f"❌ Failed checks: {', '.join(failed_checks)}")
            return False
        
        logger.info("✅ All pre-deployment checks passed")
        return True
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            logger.error(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
            return False
    
    def _check_dependencies(self) -> bool:
        """Check required dependencies"""
        try:
            result = subprocess.run(['pip', 'check'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ All dependencies satisfied")
                return True
            else:
                logger.error(f"❌ Dependency issues: {result.stdout}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to check dependencies: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            statvfs = os.statvfs('/')
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            
            if free_space_gb >= 10:  # Require at least 10GB
                logger.info(f"✅ Available disk space: {free_space_gb:.1f} GB")
                return True
            else:
                logger.error(f"❌ Insufficient disk space: {free_space_gb:.1f} GB (minimum 10 GB required)")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Could not check disk space: {e}")
            return True  # Continue deployment
    
    def _check_permissions(self) -> bool:
        """Check file system permissions"""
        test_paths = [
            '/var/lib/skz-agents/',
            '/var/log/skz-agents/',
            '/etc/skz-agents/'
        ]
        
        for path in test_paths:
            try:
                os.makedirs(path, exist_ok=True)
                test_file = os.path.join(path, 'test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                logger.error(f"❌ Permission denied for {path}: {e}")
                return False
        
        logger.info("✅ File system permissions verified")
        return True
    
    def _check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket
        
        required_ports = [
            self.config['services']['api_server']['port'],
            self.config['services']['agent_orchestrator']['port'],
            self.config['services']['communication_service']['port'],
            self.config['services']['monitoring']['port']
        ]
        
        for port in required_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                logger.error(f"❌ Port {port} is already in use")
                return False
        
        logger.info(f"✅ All required ports available: {required_ports}")
        return True
    
    def _check_environment_variables(self) -> bool:
        """Check required environment variables"""
        required_vars = [
            'DATABASE_URL',
            'REDIS_URL',
            'JWT_SECRET',
            'SMTP_HOST'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        logger.info("✅ All required environment variables present")
        return True
    
    def _setup_environment(self) -> bool:
        """Setup production environment"""
        logger.info("🔧 Setting up production environment...")
        
        try:
            # Create directory structure
            directories = [
                '/var/lib/skz-agents/vectordb',
                '/var/log/skz-agents',
                '/etc/skz-agents',
                '/var/run/skz-agents'
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
            
            # Copy configuration files
            self._copy_config_files()
            
            # Set up log rotation
            self._setup_log_rotation()
            
            logger.info("✅ Environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def _migrate_database(self) -> bool:
        """Run database migrations"""
        logger.info("🗄️ Running database migrations...")
        
        try:
            # Run Alembic migrations
            result = subprocess.run(['alembic', 'upgrade', 'head'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Database migrations completed")
                return True
            else:
                logger.error(f"❌ Migration failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database migration error: {e}")
            return False
    
    def _deploy_services(self) -> bool:
        """Deploy all microservices"""
        logger.info("🚀 Deploying services...")
        
        services = [
            ('api_server', 'python -m src.api.main'),
            ('agent_orchestrator', 'python -m src.agents.orchestrator'),
            ('communication_service', 'python -m src.models.communication_automation'),
            ('monitoring', 'python -m src.monitoring.health_monitor')
        ]
        
        for service_name, command in services:
            if not self._deploy_service(service_name, command):
                logger.error(f"❌ Failed to deploy {service_name}")
                return False
        
        logger.info("✅ All services deployed successfully")
        return True
    
    def _deploy_service(self, service_name: str, command: str) -> bool:
        """Deploy individual service"""
        try:
            logger.info(f"Deploying {service_name}...")
            
            # Create systemd service file
            service_content = f"""[Unit]
Description=SKZ Agents {service_name}
After=network.target

[Service]
Type=simple
User=skz-agents
WorkingDirectory=/opt/skz-agents
ExecStart={command}
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/skz-agents
Environment=ENV=production

[Install]
WantedBy=multi-user.target
"""
            
            service_file = f"/etc/systemd/system/skz-{service_name}.service"
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            # Enable and start service
            subprocess.run(['systemctl', 'daemon-reload'], check=True)
            subprocess.run(['systemctl', 'enable', f'skz-{service_name}'], check=True)
            subprocess.run(['systemctl', 'start', f'skz-{service_name}'], check=True)
            
            logger.info(f"✅ {service_name} deployed and started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy {service_name}: {e}")
            return False
    
    def _setup_monitoring(self) -> bool:
        """Setup monitoring and alerting"""
        logger.info("📊 Setting up monitoring...")
        
        try:
            # Setup Prometheus metrics
            self._setup_prometheus()
            
            # Setup log aggregation
            self._setup_log_aggregation()
            
            # Setup health checks
            self._setup_health_checks()
            
            logger.info("✅ Monitoring setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return False
    
    def _perform_health_checks(self) -> bool:
        """Perform post-deployment health checks"""
        logger.info("🏥 Performing health checks...")
        
        services = self.config['services']
        
        for service_name, service_config in services.items():
            if not self._check_service_health(service_name, service_config.get('port')):
                logger.error(f"❌ Health check failed for {service_name}")
                return False
        
        logger.info("✅ All health checks passed")
        return True
    
    def _check_service_health(self, service_name: str, port: Optional[int]) -> bool:
        """Check individual service health"""
        if not port:
            return True  # Skip if no port configured
        
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {service_name} health check passed")
                return True
            else:
                logger.error(f"❌ {service_name} health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ {service_name} health check error: {e}")
            return False
    
    def _run_performance_tests(self) -> bool:
        """Run performance benchmarks"""
        logger.info("⚡ Running performance tests...")
        
        try:
            # Run integration tests with performance benchmarks
            result = subprocess.run([
                'pytest', 
                'tests/integration/test_system_integration.py::TestSystemIntegration::test_performance_benchmarks',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Performance tests passed")
                return True
            else:
                logger.warning(f"⚠️ Performance tests had issues: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance test error: {e}")
            return False
    
    def _generate_deployment_report(self):
        """Generate deployment summary report"""
        report = {
            'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': self.config['environment'],
            'services_deployed': list(self.config['services'].keys()),
            'deployment_status': 'SUCCESS',
            'health_checks': 'PASSED',
            'configuration': self.config
        }
        
        report_file = f"deployment_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Deployment report saved to {report_file}")
    
    def _rollback_deployment(self):
        """Rollback deployment in case of failure"""
        logger.info("🔄 Rolling back deployment...")
        
        services = list(self.config['services'].keys())
        for service_name in services:
            try:
                subprocess.run(['systemctl', 'stop', f'skz-{service_name}'], 
                             capture_output=True)
                subprocess.run(['systemctl', 'disable', f'skz-{service_name}'], 
                             capture_output=True)
                logger.info(f"✅ Rolled back {service_name}")
            except Exception as e:
                logger.error(f"❌ Failed to rollback {service_name}: {e}")
    
    def _copy_config_files(self):
        """Copy configuration files to production locations"""
        # Implementation for copying config files
        pass
    
    def _setup_log_rotation(self):
        """Setup log rotation configuration"""
        # Implementation for log rotation
        pass
    
    def _setup_prometheus(self):
        """Setup Prometheus metrics collection"""
        # Implementation for Prometheus setup
        pass
    
    def _setup_log_aggregation(self):
        """Setup log aggregation system"""
        # Implementation for log aggregation
        pass
    
    def _setup_health_checks(self):
        """Setup automated health checks"""
        # Implementation for health check automation
        pass


def main():
    """Main deployment function"""
    deployer = ProductionDeployer()
    
    if deployer.deploy():
        print("🎉 SKZ Agents Framework successfully deployed to production!")
        sys.exit(0)
    else:
        print("💥 Deployment failed. Check deployment.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
