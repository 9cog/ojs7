<?php

/**
 * @file plugins/generic/skzAgents/classes/SKZAuthHandler.inc.php
 *
 * SKZ Authentication Handler - Handles JWT token generation and authentication for agents
 */

import('lib.pkp.classes.handler.APIHandler');
import('classes.core.Services');

class SKZAuthHandler extends APIHandler {

    /**
     * Constructor
     */
    public function __construct() {
        $this->_handlerPath = 'skzauth';
        $this->_endpoints = [
            'POST' => [
                [
                    'pattern' => $this->getEndpointPattern() . '/token',
                    'handler' => [$this, 'generateToken'],
                    'roles' => [ROLE_ID_SITE_ADMIN, ROLE_ID_MANAGER, ROLE_ID_SUB_EDITOR]
                ],
                [
                    'pattern' => $this->getEndpointPattern() . '/validate',
                    'handler' => [$this, 'validateToken'],
                    'roles' => [ROLE_ID_SITE_ADMIN, ROLE_ID_MANAGER, ROLE_ID_SUB_EDITOR]
                ]
            ]
        ];
        parent::__construct();
    }

    /**
     * @copydoc PKPHandler::authorize()
     */
    function authorize($request, &$args, $roleAssignments) {
        import('lib.pkp.classes.security.authorization.ContextAccessPolicy');
        $this->addPolicy(new ContextAccessPolicy($request, $roleAssignments));
        return parent::authorize($request, $args, $roleAssignments);
    }

    /**
     * Generate JWT token for agent authentication
     * @param $slimRequest Request Slim request object
     * @param $response Response object
     * @param $args array arguments
     * @return Response
     */
    public function generateToken($slimRequest, $response, $args) {
        $request = $this->getRequest();
        $context = $request->getContext();
        $user = $request->getUser();

        if (!$context || !$user) {
            return $response->withStatus(404)->withJsonError('api.404.resourceNotFound');
        }

        // Get user roles
        $userGroupDao = DAORegistry::getDAO('UserGroupDAO');
        $userGroups = $userGroupDao->getByUserId($user->getId(), $context->getId());
        $roles = [];
        
        while ($userGroup = $userGroups->next()) {
            $roleId = $userGroup->getRoleId();
            switch($roleId) {
                case ROLE_ID_SITE_ADMIN:
                    $roles[] = 'ROLE_ID_SITE_ADMIN';
                    break;
                case ROLE_ID_MANAGER:
                    $roles[] = 'ROLE_ID_MANAGER';
                    break;
                case ROLE_ID_SUB_EDITOR:
                    $roles[] = 'ROLE_ID_SUB_EDITOR';
                    break;
                case ROLE_ID_REVIEWER:
                    $roles[] = 'ROLE_ID_REVIEWER';
                    break;
                case ROLE_ID_AUTHOR:
                    $roles[] = 'ROLE_ID_AUTHOR';
                    break;
            }
        }

        // Create JWT payload
        $payload = [
            'user_id' => $user->getId(),
            'username' => $user->getUsername(),
            'email' => $user->getEmail(),
            'roles' => $roles,
            'context_id' => $context->getId(),
            'iat' => time(),
            'exp' => time() + (24 * 60 * 60), // 24 hours
            'iss' => 'ojs-skz'
        ];

        // Generate JWT token
        $secret = Config::getVar('security', 'skz_jwt_secret', 'default-secret-change-in-production');
        $token = $this->_generateJWT($payload, $secret);

        return $response->withJson([
            'token' => $token,
            'user' => [
                'id' => $user->getId(),
                'username' => $user->getUsername(),
                'email' => $user->getEmail(),
                'roles' => $roles
            ],
            'context_id' => $context->getId(),
            'expires_in' => 86400 // 24 hours
        ]);
    }

    /**
     * Validate JWT token
     * @param $slimRequest Request Slim request object
     * @param $response Response object
     * @param $args array arguments
     * @return Response
     */
    public function validateToken($slimRequest, $response, $args) {
        $requestData = $slimRequest->getParsedBody();
        $token = $requestData['token'] ?? null;

        if (!$token) {
            return $response->withStatus(400)->withJsonError('api.400.missingToken');
        }

        $secret = Config::getVar('security', 'skz_jwt_secret', 'default-secret-change-in-production');
        $payload = $this->_validateJWT($token, $secret);

        if (!$payload) {
            return $response->withStatus(401)->withJsonError('api.401.invalidToken');
        }

        return $response->withJson([
            'valid' => true,
            'payload' => $payload
        ]);
    }

    /**
     * Generate JWT token
     * @param array $payload
     * @param string $secret
     * @return string
     */
    private function _generateJWT($payload, $secret) {
        $header = json_encode(['typ' => 'JWT', 'alg' => 'HS256']);
        $payload = json_encode($payload);
        
        $base64Header = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($header));
        $base64Payload = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($payload));
        
        $signature = hash_hmac('sha256', $base64Header . "." . $base64Payload, $secret, true);
        $base64Signature = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($signature));
        
        return $base64Header . "." . $base64Payload . "." . $base64Signature;
    }

    /**
     * Validate JWT token
     * @param string $token
     * @param string $secret
     * @return array|null
     */
    private function _validateJWT($token, $secret) {
        $parts = explode('.', $token);
        if (count($parts) !== 3) {
            return null;
        }

        list($header, $payload, $signature) = $parts;
        
        // Verify signature
        $expectedSignature = hash_hmac('sha256', $header . "." . $payload, $secret, true);
        $expectedSignature = str_replace(['+', '/', '='], ['-', '_', ''], base64_encode($expectedSignature));
        
        if ($signature !== $expectedSignature) {
            return null;
        }

        // Decode payload
        $payload = base64_decode(str_replace(['-', '_'], ['+', '/'], $payload));
        $data = json_decode($payload, true);

        // Check expiration
        if (isset($data['exp']) && $data['exp'] < time()) {
            return null;
        }

        return $data;
    }
}