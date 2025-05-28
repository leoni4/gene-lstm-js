/* eslint-disable no-undef,@typescript-eslint/no-var-requires */
import path from 'path';
import { fileURLToPath } from 'url';

// Convert `import.meta.url` to a path
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default [
    // Configuration for UMD (for CDN usage)
    {
        entry: './src/index.ts', // Adjust to your entry file
        output: {
            filename: 'neatjs.umd.js',
            path: path.resolve(__dirname, 'dist'),
            library: 'neatjs', // Global variable name for CDN usage
            libraryTarget: 'umd',
            globalObject: 'this', // Compatibility for Node.js and browser
        },
        resolve: {
            extensions: ['.ts', '.tsx', '.js'], // Resolve TypeScript and JavaScript files
        },
        mode: 'production',
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    exclude: /node_modules/,
                    use: 'ts-loader', // Use ts-loader for TypeScript files
                },
            ],
        },
    },
    // Configuration for ES module (for modern bundlers)
    {
        entry: './src/index.ts', // Adjust to your entry file
        output: {
            filename: 'neatjs.esm.js',
            path: path.resolve(__dirname, 'dist'),
            library: {
                type: 'module',
            },
            module: true, // Enables the output to be treated as an ES module
        },
        resolve: {
            extensions: ['.ts', '.tsx', '.js'], // Allows importing these file extensions without specifying them
        },
        mode: 'production', // Use 'production' mode for optimizations in output
        experiments: {
            outputModule: true, // Required to allow module type output
        },
        module: {
            rules: [
                {
                    test: /\.tsx?$/, // Matches TypeScript files
                    exclude: /node_modules/, // Excludes dependencies from processing
                    use: 'ts-loader', // Use ts-loader to compile TypeScript
                },
            ],
        },
        externals: {
            // Specify any dependencies here if you want them to be excluded from the bundle
        },
    },
];
