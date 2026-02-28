import { defineConfig } from 'vite';

export default defineConfig({
    root: './demo',
    build: {
        outDir: '../dist-demo',
        emptyOutDir: true,
    },
    base: process.env.NODE_ENV === 'production' ? '/gene-lstm-js/' : '/',
    server: {
        port: 3000,
        open: true,
    },
    resolve: {
        alias: {
            '@': '/src',
        },
    },
});
