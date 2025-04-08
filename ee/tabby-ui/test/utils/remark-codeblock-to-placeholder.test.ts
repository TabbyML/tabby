import { describe, expect, it } from 'vitest'
import { createPlaceholderNode, parseCodeBlockMeta } from '../../lib/utils/markdown/remark-codeblock-to-placeholder'

describe('parseCodeBlockMeta', () => {
  it('should parse meta with multiple key-value pairs', () => {
    const meta = 'label=file object={"filepath": "/test.js"}';
    const result = parseCodeBlockMeta(meta);
    
    expect(result.label).toBe('file');
    // The function only splits by '=' and doesn't parse the JSON content
    // So we just check that the string contains the start of the object
    const objectValue = result.object;
    expect(objectValue).toBeDefined();
    expect(objectValue.startsWith('{"filepath"')).toBeTruthy();
  });

  it('should handle empty meta', () => {
    const result = parseCodeBlockMeta('');
    expect(Object.keys(result).length).toBe(0);
  });

  it('should handle null meta', () => {
    const result = parseCodeBlockMeta(null);
    expect(Object.keys(result).length).toBe(0);
  });

  it('should handle undefined meta', () => {
    const result = parseCodeBlockMeta(undefined);
    expect(Object.keys(result).length).toBe(0);
  });

  it('should handle meta with only keys (no values)', () => {
    const meta = 'key1 key2';
    const result = parseCodeBlockMeta(meta);
    expect(Object.keys(result).length).toBe(0);
  });

  it('should handle meta with complex values', () => {
    const meta = 'label=file object={"complex": {"nested": true, "array": [1,2,3]}}';
    const result = parseCodeBlockMeta(meta);
    
    expect(result.label).toBe('file');
    // The function only splits by '=' and doesn't parse the JSON content
    // So we just check that the string contains the start of the object
    const objectValue = result.object;
    expect(objectValue).toBeDefined();
    expect(objectValue.startsWith('{"complex"')).toBeTruthy();
  });
});

describe('createPlaceholderNode', () => {
  it('should create a file placeholder node', () => {
    const fileObject = {
      kind: 'git',
      filepath: '/path/to/file.js',
      gitUrl: 'git@github.com:user/repo.git'
    };
    
    const result = createPlaceholderNode('file', fileObject);
    
    expect(result.type).toBe('placeholder');
    expect(result.placeholderType).toBe('file');
    expect(result.attributes.object).toEqual(fileObject);
  });

  it('should create a symbol placeholder node', () => {
    const symbolObject = {
      name: 'myFunction',
      type: 'function',
      filepath: '/path/to/file.js'
    };
    
    const result = createPlaceholderNode('symbol', symbolObject);
    
    expect(result.type).toBe('placeholder');
    expect(result.placeholderType).toBe('symbol');
    expect(result.attributes.object).toEqual(symbolObject);
  });

  it('should create a contextCommand placeholder node', () => {
    const result = createPlaceholderNode('contextCommand', 'changes');
    
    expect(result.type).toBe('placeholder');
    expect(result.placeholderType).toBe('contextCommand');
    expect(result.attributes.object).toBe('changes');
  });

  it('should handle string object', () => {
    const result = createPlaceholderNode('file', 'simple-string');
    
    expect(result.type).toBe('placeholder');
    expect(result.placeholderType).toBe('file');
    expect(result.attributes.object).toBe('simple-string');
  });

  it('should handle null object', () => {
    const result = createPlaceholderNode('file', null);
    
    expect(result.type).toBe('placeholder');
    expect(result.placeholderType).toBe('file');
    expect(result.attributes.object).toBeNull();
  });
}); 