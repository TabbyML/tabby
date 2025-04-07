import { describe, expect, it } from 'vitest'
import {
  formatObjectToMarkdownBlock,
  escapeBackslashes,
  formatPlaceholder,
  customAstToString,
  processContextCommand
} from '../../lib/utils/markdown'
import { remark } from 'remark'
import remarkStringify from 'remark-stringify'

describe('escapeBackslashes', () => {
  it('should escape backslashes in Windows paths', () => {
    const windowsPath = 'C:\\Users\\test\\Documents\\project\\file.js'
    const expected = 'C:\\\\Users\\\\test\\\\Documents\\\\project\\\\file.js'
    expect(escapeBackslashes(windowsPath)).toBe(expected)
  })

  it('should handle unix paths properly', () => {
    const unixPath = '/home/user/projects/file.js'
    expect(escapeBackslashes(unixPath)).toBe(unixPath)
  })

  it('should handle empty strings', () => {
    expect(escapeBackslashes('')).toBe('')
  })

  it('should handle null or undefined', () => {
    expect(escapeBackslashes(null as any)).toBe('')
    expect(escapeBackslashes(undefined as any)).toBe('')
  })
})

describe('formatPlaceholder', () => {
  it('should format a file placeholder with Windows path', () => {
    const objStr = JSON.stringify({ filepath: 'C:\\Users\\test\\file.js' })
    const expected = `[[file:${JSON.stringify({ filepath: 'C:\\\\Users\\\\test\\\\file.js' })}]]`
    expect(formatPlaceholder('file', objStr)).toBe(expected)
  })

  it('should format a symbol placeholder with Unix path', () => {
    const objStr = JSON.stringify({ filepath: '/home/user/file.js', range: { start: 1, end: 5 } })
    expect(formatPlaceholder('symbol', objStr)).toBe(`[[symbol:${objStr}]]`)
  })

  it('should handle empty object string', () => {
    expect(formatPlaceholder('file', '')).toBe('')
  })

  it('should handle null or undefined', () => {
    expect(formatPlaceholder('file', null as any)).toBe('')
    expect(formatPlaceholder('symbol', undefined as any)).toBe('')
  })
})

describe('formatObjectToMarkdownBlock', () => {
  it('should format object with JavaScript content', () => {
    const obj = { filepath: '/path/to/file.js' }
    const jsContent = `function test() {
  console.log("Hello World");
  return true;
}`
    const result = formatObjectToMarkdownBlock('file', obj, jsContent)
    expect(result).toContain('```context label=file')
    expect(result).toContain(JSON.stringify(obj))
    expect(result).toContain(jsContent)
  })

  it('should format object with Rust content', () => {
    const obj = { filepath: 'C:\\path\\to\\file.rs' }
    const rustContent = `fn main() {
    println!("Hello, world!");
    let x = 5;
    let y = 10;
    println!("x + y = {}", x + y);
}`
    const result = formatObjectToMarkdownBlock('file', obj, rustContent)
    expect(result).toContain('```context label=file')
    expect(result).toContain(JSON.stringify(obj))
    expect(result).toContain(rustContent)
  })

  it('should handle content with markdown code blocks', () => {
    const obj = { filepath: '/path/to/README.md' }
    const markdownContent = `# Title
Some text here
\`\`\`js
const x = 5;
\`\`\`
More text`
    const result = formatObjectToMarkdownBlock('file', obj, markdownContent)
    expect(result).toContain('```context label=file')
    // Verify backticks are properly handled
    expect(result).not.toContain('```js')
    expect(result).toContain('` ` `js')
  })

  it('should handle content ending with backtick', () => {
    const obj = { filepath: '/path/to/file.md' }
    const content = 'This is some text with a backtick at the end: `'
    const result = formatObjectToMarkdownBlock('file', obj, content)
    // Should append a space after the backtick
    expect(result.includes('backtick at the end: ` \n```')).toBeTruthy()
  })

  it('should handle complex nested code blocks', () => {
    const obj = { filepath: '/path/to/doc.md' }
    const complexContent = `# Documentation
\`\`\`html
<div>
  <pre>
    \`\`\`typescript
    function test() {
      return true;
    }
    \`\`\`
  </pre>
</div>
\`\`\`
`
    const result = formatObjectToMarkdownBlock('file', obj, complexContent)
    // Check that nested code blocks are handled properly
    expect(result).toContain('` ` `html')
    expect(result).toContain('` ` `typescript')
  })

  it('should handle Windows paths in object properties', () => {
    const obj = { filepath: 'C:\\Users\\test\\file.txt' }
    const content = 'Simple text content'
    const result = formatObjectToMarkdownBlock('file', obj, content)
    expect(result).toContain(JSON.stringify(obj))
    expect(result).toContain(content)
  })

  it('should handle error in object serialization', () => {
    // Create a circular reference that will cause JSON.stringify to fail
    const circularObj: any = { name: 'test' }
    circularObj.self = circularObj
    
    const content = 'Some content'
    const result = formatObjectToMarkdownBlock('file', circularObj, content)
    expect(result).toBe('\n*Error formatting file*\n')
  })
})

describe('customAstToString', () => {
  it('should properly stringify a simple markdown AST', () => {
    // 使用标准的remark处理器解析文本
    const markdownText = '# Title\n\nThis is a paragraph.\n\n* List item 1\n* List item 2'
    const ast = remark().parse(markdownText)
    
    const result = customAstToString(ast)
    
    // 验证最基本的结构被保留
    expect(result).toContain('# Title')
    expect(result).toContain('This is a paragraph')
    expect(result).toContain('* List item 1')
    expect(result).toContain('* List item 2')
  })
  
  it('should preserve code blocks in the AST', () => {
    // 使用标准的remark处理器解析文本
    const markdownText = '```js\nconst x = 5;\n```'
    const ast = remark().parse(markdownText)
    
    const result = customAstToString(ast)
    
    expect(result).toContain('```js')
    expect(result).toContain('const x = 5;')
    expect(result).toContain('```')
  })
  
  it('should maintain processContextCommand functionality', () => {
    const input = '```context label=file object={"filepath":"/path/to/file.js"}\nconst x = 5;\n```'
    const result = processContextCommand(input)
    
    expect(result).toContain('[[file:')
    expect(result).toContain('{"filepath":"/path/to/file.js"}')
  })
})
