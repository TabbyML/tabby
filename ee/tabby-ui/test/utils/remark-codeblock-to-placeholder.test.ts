import { describe, expect, it } from 'vitest'

import {
  createPlaceholderNode,
  parseCodeBlockMeta
} from '../../lib/utils/markdown/remark-codeblock-to-placeholder'

describe('parseCodeBlockMeta', () => {
  it('should parse meta with multiple key-value pairs', () => {
    const meta = 'label=file object={"filepath": "/test.js"}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    // The function only splits by '=' and doesn't parse the JSON content
    // So we just check that the string contains the start of the object
    const objectValue = result.object
    expect(objectValue).toBeDefined()
    expect(objectValue.startsWith('{"filepath"')).toBeTruthy()
  })

  it('should handle empty meta', () => {
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

  it('should handle meta with only keys (no values)', () => {
    const meta = 'key1 key2'
    const result = parseCodeBlockMeta(meta)
    expect(Object.keys(result).length).toBe(0)
  })

  it('should handle meta with complex values', () => {
    const meta =
      'label=file object={"complex": {"nested": true, "array": [1,2,3]}}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    // The function only splits by '=' and doesn't parse the JSON content
    // So we just check that the string contains the start of the object
    const objectValue = result.object
    expect(objectValue).toBeDefined()
    expect(objectValue.startsWith('{"complex"')).toBeTruthy()
  })

  // Tests for the fixed bug with complex nested objects containing spaces
  it('should correctly parse meta with nested objects containing spaces', () => {
    // A complex meta with nested objects and space in the object value
    const meta =
      'label=symbol object={"filepath":{"kind":"git","filepath":"CODE_OF_CONDUCT.md","gitUrl":"https://github.com/TabbyML/tabby"},"range":{"start":1,"end":15},"label":"# Contributor Covenant Code of Conduct"}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('symbol')
    const objectValue = result.object
    expect(objectValue).toBeDefined()

    // Check for various parts of the complex object structure
    expect(objectValue).toContain('filepath')
    expect(objectValue).toContain('CODE_OF_CONDUCT.md')
    expect(objectValue).toContain('TabbyML/tabby')
    expect(objectValue).toContain('Contributor Covenant Code of Conduct')
  })

  it('should handle meta with quoted values containing spaces', () => {
    // Meta with quoted values containing spaces
    const meta =
      'label=file description="This is a file with spaces" object={"name":"test file.js"}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    expect(result.description).toBeDefined()
    expect(result.description).toContain('This is a file with spaces')
    expect(result.object).toContain('test file.js')
  })

  it('should correctly parse meta with multiple complex values and spaces', () => {
    // Test case with multiple complex values and spaces
    const meta =
      'label=file object={"path": "/some/path with spaces/file.js"} extra={"data": "value with spaces", "num": 123}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    expect(result.object).toContain('path with spaces')
    expect(result.extra).toContain('value with spaces')
    expect(result.extra).toContain('123')
  })

  it('should parse meta with escaped quotes in values', () => {
    // Test with escaped quotes in JSON strings
    const meta = 'label=symbol object={"text":"Some \\"quoted\\" content"}'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('symbol')
    expect(result.object).toContain('Some \\"quoted\\" content')
  })

  it('should handle meta with equals sign in the value', () => {
    // Test with equals sign in the value
    const meta = 'label=file object={"equation":"x=y+z"} condition="if (a==b)"'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('file')
    // Object with equals sign in value should be properly parsed
    expect(result.object).toBeDefined()
    // Check for condition parameter which should be parsed correctly
    expect(result.condition).toBe('if (a==b)')
  })

  it('should handle complex meta with multiple attributes and quoted labels', () => {
    // Test with complex object, equals in value, and quoted label with spaces
    const meta = 'label=file object={haha:"x=y"} label="#nihao explain doc"'
    const result = parseCodeBlockMeta(meta)

    expect(result.label).toBe('#nihao explain doc')
    expect(result.object).toBeDefined()
    expect(result.object).toContain('haha')
    expect(result.object).toContain('x=y')
  })
})

describe('createPlaceholderNode', () => {
  it('should create a file placeholder node', () => {
    const fileObject = {
      kind: 'git',
      filepath: '/path/to/file.js',
      gitUrl: 'git@github.com:user/repo.git'
    }

    const result = createPlaceholderNode('file', fileObject)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toEqual(fileObject)
  })

  it('should create a symbol placeholder node', () => {
    const symbolObject = {
      name: 'myFunction',
      type: 'function',
      filepath: '/path/to/file.js'
    }

    const result = createPlaceholderNode('symbol', symbolObject)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('symbol')
    expect(result.attributes.object).toEqual(symbolObject)
  })

  it('should create a contextCommand placeholder node', () => {
    const result = createPlaceholderNode('contextCommand', 'changes')

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('contextCommand')
    expect(result.attributes.object).toBe('changes')
  })

  it('should handle string object', () => {
    const result = createPlaceholderNode('file', 'simple-string')

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toBe('simple-string')
  })

  it('should handle null object', () => {
    const result = createPlaceholderNode('file', null)

    expect(result.type).toBe('placeholder')
    expect(result.placeholderType).toBe('file')
    expect(result.attributes.object).toBeNull()
  })
})
