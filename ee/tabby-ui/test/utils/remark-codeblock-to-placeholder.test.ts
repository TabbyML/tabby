import { describe, expect, it } from 'vitest'

import {
  createPlaceholderNode,
  parseCodeBlockMeta
} from '../../lib/utils/markdown/remark-codeblock-to-placeholder'

describe('parseCodeBlockMeta', () => {
  it('should parse meta string containing JSON object', () => {
    const meta = '{"label":"file", "object":{"filepath": "/test.js"}}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    expect(result.object).toEqual({ filepath: '/test.js' }) // Expect parsed object
  })

  it('should handle empty meta string', () => {
    const result = parseCodeBlockMeta('')
    expect(Object.keys(result).length).toBe(0)
  })

  it('should handle null meta', () => {
    const result = parseCodeBlockMeta(null)
    expect(Object.keys(result).length).toBe(0)
  })

  it('should handle undefined meta', () => {
    const result = parseCodeBlockMeta(undefined)
    expect(Object.keys(result).length).toBe(0)
  })

  it('should handle invalid JSON meta string', () => {
    const meta = 'label=file object={invalid json' // Invalid JSON
    const result = parseCodeBlockMeta(meta)
    // Should return empty object on parse error
    expect(Object.keys(result).length).toBe(0)
  })

  it('should parse complex JSON object in meta', () => {
    const meta =
      '{"label":"file", "object":{"complex": {"nested": true, "array": [1,2,3]}}}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    expect(result.object).toEqual({
      complex: { nested: true, array: [1, 2, 3] }
    })
  })

  it('should correctly parse nested objects from JSON meta', () => {
    const meta =
      '{"label":"symbol", "object":{"filepath":{"kind":"git","filepath":"CODE_OF_CONDUCT.md","gitUrl":"https://github.com/TabbyML/tabby"},"range":{"start":1,"end":15},"label":"# Contributor Covenant Code of Conduct"}}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('symbol')
    expect(result.object).toEqual({
      filepath: {
        kind: 'git',
        filepath: 'CODE_OF_CONDUCT.md',
        gitUrl: 'https://github.com/TabbyML/tabby'
      },
      range: { start: 1, end: 15 },
      label: '# Contributor Covenant Code of Conduct'
    })
  })

  // Removed tests that relied on the old key=value parsing logic
  // as the function now expects a single JSON string.
})

describe('createPlaceholderNode', () => {
  it('should create a file placeholder node with stringified object', () => {
    const fileObject = {
      kind: 'git',
      filepath: '/path/to/file.js',
      gitUrl: 'git@github.com:user/repo.git'
    }
    const fileObjectString = JSON.stringify(fileObject)

    const result = createPlaceholderNode('file', fileObjectString) // Pass stringified object

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toBe(fileObjectString) // Expect the string back
  })

  it('should create a symbol placeholder node with stringified object', () => {
    const symbolObject = {
      name: 'myFunction',
      type: 'function',
      filepath: '/path/to/file.js'
    }
    const symbolObjectString = JSON.stringify(symbolObject)

    const result = createPlaceholderNode('symbol', symbolObjectString) // Pass stringified object

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('symbol')
    expect(result.attributes.object).toBe(symbolObjectString) // Expect the string back
  })

  it('should create a contextCommand placeholder node', () => {
    const command = 'changes'
    const result = createPlaceholderNode('contextCommand', command)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('contextCommand')
    expect(result.attributes.object).toBe(command)
  })

  it('should handle simple string object', () => {
    const simpleString = 'simple-string'
    const result = createPlaceholderNode('file', simpleString)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toBe(simpleString)
  })

  it('should handle stringified null object', () => {
    const nullString = JSON.stringify(null) // Pass 'null' as a string
    const result = createPlaceholderNode('file', nullString)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toBe(nullString) // Expect 'null' string back
  })
})
