#!/usr/bin/env node

const fs = require('fs-extra');
const path = require('path');

const cwd = process.cwd();
const sourceDir = path.join(cwd, 'node_modules', 'tabby-agent', 'dist', 'node');
const targetDir = path.join(cwd, 'plugin', 'tabby-agent', 'dist', 'node');

async function copyFiles() {
  try {
    await fs.emptyDir(targetDir);
    await fs.copy(sourceDir, targetDir, {
      filter: (src) => !src.endsWith('.js.map')
    });
    console.log('✅ Files copied: node_modules/tabby-agent/dist/node -> plugin/tabby-agent/dist/node');
  } catch (err) {
    console.error('❌ Error copying files:', err);
  }
}

copyFiles();
