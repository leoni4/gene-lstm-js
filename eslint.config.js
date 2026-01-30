import eslint from '@eslint/js';
import tseslint from '@typescript-eslint/eslint-plugin';
import tsparser from '@typescript-eslint/parser';
import eslintConfigPrettier from 'eslint-config-prettier';
import globals from 'globals';

export default [
    // Global ignores
    {
        ignores: [
            'out/**',
            'dist/**',
            'logs/**',
            '*.log',
            'npm-debug.log*',
            'yarn-debug.log*',
            'yarn-error.log*',
            'pids/**',
            '*.pid',
            '*.seed',
            '*.pid.lock',
            'lib-cov/**',
            'coverage/**',
            '.nyc_output/**',
            '.grunt/**',
            'bower_components/**',
            '.lock-wscript/**',
            'build/Release/**',
            'node_modules/**',
            'jspm_packages/**',
            'typings/**',
            '.npm/**',
            '.eslintcache',
            '.node_repl_history',
            '*.tgz',
            '.yarn-integrity',
            '.env',
            '.cache/**',
            '.serverless/**',
            '.idea/**',
            'sw.*',
            '.DS_Store',
            '*.swp',
            'index.js',
        ],
    },
    // Base ESLint recommended rules
    eslint.configs.recommended,
    // Configuration for TypeScript files
    {
        files: ['**/*.ts', '**/*.tsx'],
        languageOptions: {
            parser: tsparser,
            parserOptions: {
                ecmaVersion: 'latest',
                sourceType: 'module',
            },
            globals: {
                ...globals.browser,
                ...globals.es2021,
                ...globals.node,
            },
        },
        plugins: {
            '@typescript-eslint': tseslint,
        },
        rules: {
            ...tseslint.configs.recommended.rules,
            '@typescript-eslint/no-this-alias': 0,
            'padding-line-between-statements': ['error', { blankLine: 'always', prev: '*', next: 'return' }],
        },
    },
    // Prettier config (must be last to override other configs)
    eslintConfigPrettier,
];
