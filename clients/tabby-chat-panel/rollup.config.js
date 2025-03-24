import typescript from '@rollup/plugin-typescript'
import commonjs from '@rollup/plugin-commonjs'
import resolve from '@rollup/plugin-node-resolve'
import terser from '@rollup/plugin-terser'
import { defineConfig } from 'rollup'

export default defineConfig([{
  input: 'src/browser.ts',
  output: {
    dir: 'dist',
    format: 'iife',
    entryFileNames: 'iife/tabby-chat-panel.min.js',
    name: 'TabbyChatPanel',
  },
  treeshake: true,
  plugins: [
    resolve({
      browser: true,
    }),
    commonjs(),
    terser(),
    typescript({
      tsconfig: './tsconfig.json',
      noEmitOnError: true,
    }),
  ],
}])
