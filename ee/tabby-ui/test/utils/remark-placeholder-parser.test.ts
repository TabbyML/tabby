import { describe, expect, it } from 'vitest'
import { parsePlaceholder, transformPlaceholders } from '../../lib/utils/markdown/remark-placeholder-parser'

describe('parsePlaceholder', () => {
  it('should parse source placeholder correctly', () => {
    const text = '[[source:github:E12n3q]]';
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('source');
    expect(result?.placeholderNode.attributes.sourceId).toBe('github:E12n3q');
  });

  it('should parse file placeholder correctly', () => {
    const fileInfo = {
      kind: 'git',
      filepath: '/path/to/file.js',
      gitUrl: 'git@github.com:user/repo.git'
    };
    const text = `[[file:${JSON.stringify(fileInfo)}]]`;
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('file');
    
    // Parse the JSON string back to an object for comparison
    const parsedObject = JSON.parse(result?.placeholderNode.attributes.object);
    expect(parsedObject).toEqual(fileInfo);
  });

  it('should parse file placeholder with Windows path correctly', () => {
    const fileInfo = {
      kind: 'uri',
      uri: 'C:\\Users\\user\\file.js'
    };
    const text = `[[file:${JSON.stringify(fileInfo)}]]`;
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('file');
    
    // Parse the JSON string back to an object for comparison
    const parsedObject = JSON.parse(result?.placeholderNode.attributes.object);
    expect(parsedObject.kind).toBe('uri');
    expect(parsedObject.uri.includes('C:')).toBe(true);
    expect(parsedObject.uri.includes('file.js')).toBe(true);
  });

  it('should parse symbol placeholder correctly', () => {
    const symbolInfo = {
      name: 'myFunction',
      type: 'function',
      filepath: '/path/to/file.js'
    };
    const text = `[[symbol:${JSON.stringify(symbolInfo)}]]`;
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('symbol');
    
    // Parse the JSON string back to an object for comparison
    const parsedObject = JSON.parse(result?.placeholderNode.attributes.object);
    expect(parsedObject).toEqual(symbolInfo);
  });

  it('should parse contextCommand placeholder correctly', () => {
    const text = '[[contextCommand:changes]]';
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('contextCommand');
    expect(result?.placeholderNode.attributes.command).toBe('changes');
  });

  it('should handle placeholder without colon correctly', () => {
    const text = '[[simpleplaceholder]]';
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('undefined');
    expect(result?.placeholderNode.attributes.content).toBe('[[simpleplaceholder]]');
  });

  it('should handle unknown prefix correctly', () => {
    const text = '[[unknown:value]]';
    const result = parsePlaceholder(text);
    
    expect(result).not.toBeNull();
    expect(result?.matchLength).toBe(text.length);
    expect(result?.placeholderNode.type).toBe('placeholder');
    expect(result?.placeholderNode.placeholderType).toBe('undefined');
    expect(result?.placeholderNode.attributes.content).toBe('[[unknown:value]]');
  });

  it('should return null for invalid placeholder syntax', () => {
    const text = 'not a placeholder';
    const result = parsePlaceholder(text);
    
    expect(result).toBeNull();
  });

  it('should handle incomplete placeholder syntax', () => {
    const text = '[[incomplete';
    const result = parsePlaceholder(text);
    
    expect(result).toBeNull();
  });
});

describe('transformPlaceholders', () => {
  it('should transform text with a source placeholder', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'paragraph',
          children: [
            { type: 'text', value: 'This is a ' },
            { type: 'text', value: '[[source:github:E12n3q]]' },
            { type: 'text', value: ' reference.' }
          ]
        }
      ]
    };
    
    const result = transformPlaceholders(tree);
    
    expect(result.children[0].children.length).toBe(3);
    expect(result.children[0].children[0].type).toBe('text');
    expect(result.children[0].children[0].value).toBe('This is a ');
    expect(result.children[0].children[1].type).toBe('placeholder');
    expect(result.children[0].children[1].placeholderType).toBe('source');
    expect(result.children[0].children[1].attributes.sourceId).toBe('github:E12n3q');
    expect(result.children[0].children[2].type).toBe('text');
    expect(result.children[0].children[2].value).toBe(' reference.');
  });

  it('should transform text with multiple placeholders', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'paragraph',
          children: [
            { type: 'text', value: 'First [[source:github:E12n3q]] and then [[contextCommand:changes]].' }
          ]
        }
      ]
    };
    
    const result = transformPlaceholders(tree);
    
    // The function splits the text and creates 5 children:
    // 1. "First "
    // 2. Placeholder for source
    // 3. " and then "
    // 4. Placeholder for contextCommand
    // 5. "."
    expect(result.children[0].children.length).toBe(5);
    expect(result.children[0].children[0].type).toBe('text');
    expect(result.children[0].children[0].value).toBe('First ');
    expect(result.children[0].children[1].type).toBe('placeholder');
    expect(result.children[0].children[1].placeholderType).toBe('source');
    expect(result.children[0].children[2].type).toBe('text');
    expect(result.children[0].children[2].value).toBe(' and then ');
    expect(result.children[0].children[3].type).toBe('placeholder');
    expect(result.children[0].children[3].placeholderType).toBe('contextCommand');
    expect(result.children[0].children[4].type).toBe('text');
    expect(result.children[0].children[4].value).toBe('.');
  });

  it('should handle invalid placeholder syntax', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'paragraph',
          children: [
            { type: 'text', value: 'Text with [[incomplete placeholder.' }
          ]
        }
      ]
    };
    
    const result = transformPlaceholders(tree);
    
    // The function splits the text into:
    // 1. "Text with "
    // 2. "[["
    // 3. "incomplete placeholder."
    expect(result.children[0].children.length).toBe(3);
    expect(result.children[0].children[0].type).toBe('text');
    expect(result.children[0].children[0].value).toBe('Text with ');
    expect(result.children[0].children[1].type).toBe('text');
    expect(result.children[0].children[1].value).toBe('[[');
    expect(result.children[0].children[2].type).toBe('text');
    expect(result.children[0].children[2].value).toBe('incomplete placeholder.');
  });

  it('should not transform non-text nodes', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'paragraph',
          children: [
            { type: 'code', value: '[[source:github:E12n3q]]' },
            { type: 'text', value: 'Normal text.' }
          ]
        }
      ]
    };
    
    const result = transformPlaceholders(tree);
    
    expect(result.children[0].children.length).toBe(2);
    expect(result.children[0].children[0].type).toBe('code');
    expect(result.children[0].children[0].value).toBe('[[source:github:E12n3q]]');
    expect(result.children[0].children[1].type).toBe('text');
    expect(result.children[0].children[1].value).toBe('Normal text.');
  });

  it('should handle nested children', () => {
    const tree = {
      type: 'root',
      children: [
        {
          type: 'blockquote',
          children: [
            {
              type: 'paragraph',
              children: [
                { type: 'text', value: 'Nested [[contextCommand:changes]].' }
              ]
            }
          ]
        }
      ]
    };
    
    const result = transformPlaceholders(tree);
    
    const paragraphNode = result.children[0].children[0];
    // The function splits the text into:
    // 1. "Nested "
    // 2. Placeholder for contextCommand
    // 3. "."
    expect(paragraphNode.children.length).toBe(3);
    expect(paragraphNode.children[0].type).toBe('text');
    expect(paragraphNode.children[0].value).toBe('Nested ');
    expect(paragraphNode.children[1].type).toBe('placeholder');
    expect(paragraphNode.children[1].placeholderType).toBe('contextCommand');
    expect(paragraphNode.children[2].type).toBe('text');
    expect(paragraphNode.children[2].value).toBe('.');
  });

  it('should return null for null input', () => {
    const result = transformPlaceholders(null);
    expect(result).toBeNull();
  });
}); 