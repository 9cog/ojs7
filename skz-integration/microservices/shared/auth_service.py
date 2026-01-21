"""
Authentication Service for SKZ Autonomous Agents
Provides JWT token generation, validation and role-based authorization
"""

import jwt
import hashlib
import hmac
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
from flask import request, jsonify, current_app
import logging

logger = logging.getLogger(__name__)

class AuthenticationService:
    """Service for handling JWT authentication and authorization"""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)  # 24 hour token expiry
        
    def generate_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': user_data.get('user_id'),
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'roles': user_data.get('roles', []),
            'context_id': user_data.get('context_id'),
            'exp': datetime.utcnow() + self.token_expiry,
            'iat': datetime.utcnow(),
            'iss': 'skz-agents'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Generated token for user {user_data.get('username')}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token expiry
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                logger.warning("Token expired")
                return None
                
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def generate_api_signature(self, data: str, timestamp: str, api_secret: str) -> str:
        """Generate HMAC signature for API requests"""
        message = f"{timestamp}:{data}"
        signature = hmac.new(
            api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def validate_api_signature(self, data: str, timestamp: str, 
                             signature: str, api_secret: str) -> bool:
        """Validate HMAC signature for API requests"""
        expected_signature = self.generate_api_signature(data, timestamp, api_secret)
        return hmac.compare_digest(signature, expected_signature)

class AuthorizationService:
    """Service for handling role-based authorization"""
    
    # OJS Role mapping to agent permissions
    ROLE_PERMISSIONS = {
        'ROLE_ID_SITE_ADMIN': [
            'agents:manage',
            'agents:view',
            'agents:execute',
            'system:admin',
            'data:read',
            'data:write',
            'analytics:view'
        ],
        'ROLE_ID_MANAGER': [
            'agents:view',
            'agents:execute',
            'data:read',
            'data:write',
            'analytics:view'
        ],
        'ROLE_ID_SUB_EDITOR': [
            'agents:view',
            'agents:execute',
            'data:read',
            'analytics:view'
        ],
        'ROLE_ID_REVIEWER': [
            'agents:view',
            'data:read'
        ],
        'ROLE_ID_AUTHOR': [
            'agents:view',
            'data:read'
        ]
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_user_permissions(self, roles: List[str]) -> List[str]:
        """Get all permissions for a user based on their roles"""
        permissions = set()
        
        for role in roles:
            role_permissions = self.ROLE_PERMISSIONS.get(role, [])
            permissions.update(role_permissions)
        
        return list(permissions)
    
    def check_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = self.get_user_permissions(user_roles)
        return required_permission in user_permissions
    
    def check_permissions(self, user_roles: List[str], required_permissions: List[str]) -> bool:
        """Check if user has all required permissions"""
        user_permissions = set(self.get_user_permissions(user_roles))
        required_permissions_set = set(required_permissions)
        return required_permissions_set.issubset(user_permissions)

# Global instances
auth_service = AuthenticationService(
    secret_key=os.getenv('SKZ_JWT_SECRET', 'skz-default-secret-change-in-production')
)
authorization_service = AuthorizationService()

def require_auth(f):
    """Decorator to require authentication for Flask routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
        
        token = auth_header.split(' ')[1]
        user_data = auth_service.validate_token(token)
        
        if not user_data:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Add user data to request context
        request.user_data = user_data
        return f(*args, **kwargs)
    
    return decorated_function

def require_permission(permission: str):
    """Decorator to require specific permission for Flask routes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request, 'user_data'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_roles = request.user_data.get('roles', [])
            
            if not authorization_service.check_permission(user_roles, permission):
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def require_permissions(permissions: List[str]):
    """Decorator to require multiple permissions for Flask routes"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request, 'user_data'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_roles = request.user_data.get('roles', [])
            
            if not authorization_service.check_permissions(user_roles, permissions):
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def get_current_user() -> Optional[Dict[str, Any]]:
    """Get current authenticated user data"""
    return getattr(request, 'user_data', None)

def validate_api_request():
    """Validate API request with signature verification"""
    api_key = request.headers.get('X-API-Key')
    timestamp = request.headers.get('X-Timestamp')
    signature = request.headers.get('X-Signature')
    
    if not all([api_key, timestamp, signature]):
        return False, "Missing required headers"
    
    # Check timestamp (prevent replay attacks)
    try:
        request_time = float(timestamp)
        current_time = time.time()
        
        # Allow 5 minutes window
        if abs(current_time - request_time) > 300:
            return False, "Request timestamp too old"
    except ValueError:
        return False, "Invalid timestamp format"
    
    # Get API secret for the key (this would normally be from database)
    api_secret = os.getenv('SKZ_API_SECRET', 'default-api-secret')
    
    # Get request data
    if request.is_json:
        data = request.get_json()
        data_str = str(data) if data else ""
    else:
        data_str = request.get_data(as_text=True)
    
    # Validate signature
    is_valid = auth_service.validate_api_signature(data_str, timestamp, signature, api_secret)
    
    if not is_valid:
        return False, "Invalid signature"
    
    return True, "Valid request"

def require_api_auth(f):
    """Decorator to require API authentication for Flask routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_valid, message = validate_api_request()
        
        if not is_valid:
            return jsonify({'error': message}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function