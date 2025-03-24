#!/usr/bin/env node

const fs = require('fs-extra');
const path = require('path');

const cwd = process.cwd();

async function copyTabbyAgentScript() {
  const sourceDir = path.join(cwd, 'node_modules', 'tabby-agent', 'dist', 'node');
  const targetDir = path.join(cwd, 'plugin', 'tabby-agent', 'dist', 'node');
  try {
    await fs.emptyDir(targetDir);
    await fs.copy(sourceDir, targetDir, {
      filter: (src) => !src.endsWith('.js.map')
    });
    console.log(`✅ Files copied: ${sourceDir} -> ${targetDir}`);
  } catch (err) {
    console.error('❌ Error copying files:', err);
  }
}

async function copyTabbyChatPanelScript() {
  const sourceFile = path.join(cwd, 'node_modules', 'tabby-chat-panel', 'dist', 'iife', 'tabby-chat-panel.min.js');
  const targetFile = path.join(cwd, 'plugin', 'chat-panel', 'tabby-chat-panel.min.js');
  try {
    await fs.copy(sourceFile, targetFile);
    console.log(`✅ Files copied: ${sourceFile} -> ${targetFile}`);
  } catch (err) {
    console.error('❌ Error copying files:', err);
  }
}

copyTabbyAgentScript();
copyTabbyChatPanelScript();
