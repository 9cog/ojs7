/**
 * Production AI Engine using node-llama-cpp
 * Provides real AI inference for SKZ autonomous agents
 * Zero tolerance for mock implementations
 */

const { LlamaModel, LlamaContext, LlamaChatSession } = require('node-llama-cpp');
const fs = require('fs').promises;
const path = require('path');
const winston = require('winston');

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'ai-engine.log' }),
        new winston.transports.Console()
    ]
});

class ProductionAIEngine {
    constructor(config = {}) {
        this.config = {
            modelPath: config.modelPath || process.env.AI_MODEL_PATH || './models/llama-2-7b-chat.q4_0.gguf',
            contextSize: parseInt(config.contextSize || process.env.AI_CONTEXT_LENGTH || '2048'),
            temperature: parseFloat(config.temperature || process.env.AI_TEMPERATURE || '0.7'),
            topP: parseFloat(config.topP || process.env.AI_TOP_P || '0.95'),
            threads: parseInt(config.threads || process.env.AI_THREADS || '4'),
            fallbackModelPath: config.fallbackModelPath || process.env.AI_FALLBACK_MODEL_PATH,
            forceProduction: config.forceProduction || process.env.AI_FORCE_PRODUCTION === 'true',
            ...config
        };
        
        this.model = null;
        this.context = null;
        this.isInitialized = false;
        
        // Production quality check
        if (process.env.NODE_ENV === 'production' || this.config.forceProduction) {
            this._validateProductionRequirements();
        }
    }
    
    _validateProductionRequirements() {
        const requirements = [
            [this.config.modelPath, `LLaMA model path not configured: ${this.config.modelPath}`],
            [this.config.contextSize > 0, 'Context size must be positive'],
            [this.config.threads > 0, 'Thread count must be positive']
        ];
        
        for (const [requirement, errorMsg] of requirements) {
            if (!requirement) {
                throw new Error(`PRODUCTION VIOLATION: ${errorMsg}. NEVER SACRIFICE QUALITY!!`);
            }
        }
    }
    
    async initialize() {
        if (this.isInitialized) {
            return;
        }
        
        logger.info('Initializing production AI engine...');
        
        try {
            await this._initializeLlamaModel();
            this.isInitialized = true;
            logger.info('✅ Production AI engine initialized successfully');
            
        } catch (error) {
            if (this.config.forceProduction) {
                throw new Error(`PRODUCTION VIOLATION: AI engine initialization failed: ${error.message}`);
            }
            
            if (this.config.fallbackModelPath) {
                logger.warn(`Primary model failed, attempting fallback: ${error.message}`);
                await this._initializeFallbackModel();
                this.isInitialized = true;
            } else {
                throw error;
            }
        }
    }
    
    async _initializeLlamaModel() {
        // Validate model file exists
        try {
            await fs.access(this.config.modelPath);
        } catch (error) {
            if (this.config.forceProduction) {
                throw new Error(`LLaMA model required but not found: ${this.config.modelPath}`);
            }
            throw new Error(`Model file not accessible: ${this.config.modelPath}`);
        }
        
        logger.info(`Loading LLaMA model: ${this.config.modelPath}`);
        
        // Initialize model
        this.model = new LlamaModel({
            modelPath: this.config.modelPath
        });
        
        // Initialize context
        this.context = new LlamaContext({
            model: this.model,
            contextSize: this.config.contextSize,
            threads: this.config.threads
        });
        
        // Test inference
        await this._testInference();
        
        logger.info('✅ LLaMA model initialized and tested successfully');
    }
    
    async _initializeFallbackModel() {
        if (!this.config.fallbackModelPath) {
            throw new Error('No fallback model configured');
        }
        
        logger.warn('Initializing fallback model - performance will be degraded');
        
        try {
            await fs.access(this.config.fallbackModelPath);
        } catch (error) {
            throw new Error(`Fallback model not accessible: ${this.config.fallbackModelPath}`);
        }
        
        this.model = new LlamaModel({
            modelPath: this.config.fallbackModelPath
        });
        
        this.context = new LlamaContext({
            model: this.model,
            contextSize: Math.min(this.config.contextSize, 1024), // Smaller context for fallback
            threads: Math.max(this.config.threads - 2, 1)
        });
        
        await this._testInference();
        logger.info('✅ Fallback AI engine initialized');
    }
    
    async _testInference() {
        const testSession = new LlamaChatSession({
            context: this.context
        });
        
        const testPrompt = 'Test prompt for AI engine validation.';
        const response = await testSession.prompt(testPrompt, {
            maxTokens: 10,
            temperature: 0.1
        });
        
        if (!response || response.length === 0) {
            throw new Error('AI engine test inference failed - no response generated');
        }
        
        logger.info('AI engine test inference successful');
    }
    
    async generateText(prompt, options = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        if (!this.context) {
            throw new Error('PRODUCTION VIOLATION: No AI context available for text generation');
        }
        
        try {
            const session = new LlamaChatSession({
                context: this.context
            });
            
            const response = await session.prompt(prompt, {
                maxTokens: options.maxTokens || 512,
                temperature: options.temperature || this.config.temperature,
                topP: options.topP || this.config.topP,
                ...options
            });
            
            if (!response || response.length === 0) {
                throw new Error('AI inference returned empty response');
            }
            
            return response;
            
        } catch (error) {
            logger.error(`Text generation failed: ${error.message}`);
            if (this.config.forceProduction) {
                throw new Error(`PRODUCTION VIOLATION: Text generation failed: ${error.message}`);
            }
            throw error;
        }
    }
    
    async classifyText(text, categories) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        // For production text classification, use the LLaMA model with structured prompts
        const classificationPrompt = `
Classify the following text into one of these categories: ${categories.join(', ')}.

Text: "${text}"

Respond with only the category name that best fits the text.`;

        try {
            const response = await this.generateText(classificationPrompt, {
                maxTokens: 50,
                temperature: 0.1
            });
            
            // Parse response and create scores
            const normalizedResponse = response.toLowerCase().trim();
            const categoryScores = {};
            
            for (const category of categories) {
                if (normalizedResponse.includes(category.toLowerCase())) {
                    categoryScores[category] = 0.8;
                } else {
                    categoryScores[category] = 0.2 / (categories.length - 1);
                }
            }
            
            // Normalize scores to sum to 1.0
            const totalScore = Object.values(categoryScores).reduce((sum, score) => sum + score, 0);
            if (totalScore > 0) {
                for (const category in categoryScores) {
                    categoryScores[category] /= totalScore;
                }
            }
            
            return categoryScores;
            
        } catch (error) {
            logger.error(`Text classification failed: ${error.message}`);
            if (this.config.forceProduction) {
                throw new Error(`PRODUCTION VIOLATION: Text classification failed: ${error.message}`);
            }
            throw error;
        }
    }
    
    async getEngineStatus() {
        return {
            initialized: this.isInitialized,
            modelPath: this.config.modelPath,
            contextSize: this.config.contextSize,
            threads: this.config.threads,
            forceProduction: this.config.forceProduction,
            modelLoaded: this.model !== null,
            contextLoaded: this.context !== null
        };
    }
    
    async shutdown() {
        if (this.context) {
            this.context = null;
        }
        if (this.model) {
            this.model = null;
        }
        this.isInitialized = false;
        logger.info('AI engine shutdown complete');
    }
}

// Global AI engine instance
let globalAIEngine = null;

async function getProductionAIEngine(config) {
    if (!globalAIEngine) {
        globalAIEngine = new ProductionAIEngine(config);
        await globalAIEngine.initialize();
    }
    return globalAIEngine;
}

module.exports = {
    ProductionAIEngine,
    getProductionAIEngine
};

// Handle graceful shutdown
process.on('SIGINT', async () => {
    if (globalAIEngine) {
        await globalAIEngine.shutdown();
    }
    process.exit(0);
});

process.on('SIGTERM', async () => {
    if (globalAIEngine) {
        await globalAIEngine.shutdown();
    }
    process.exit(0);
});