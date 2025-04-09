import { Filepath } from 'tabby-chat-panel/index'
import { describe, expect, it } from 'vitest'

import {
  formatObjectToMarkdownBlock,
  shouldAddPrefixNewline,
  shouldAddSuffixNewline
} from '../../lib/utils/markdown'

describe('formatObjectToMarkdownBlock - comprehensive tests', () => {
  describe('filepath types with standard content', () => {
    const jsContent = `// Example JavaScript code
function example() {
  const greeting = "Hello World!";
  console.log(greeting);
  return {
    message: greeting,
    timestamp: new Date().getTime()
  };
}

// ES6 features
const arrowFunc = () => {
  return Promise.resolve(42);
};
`

    it('should format Unix path with git format', () => {
      const unixGitObj = {
        kind: 'git',
        filepath: '/home/user/projects/example.js',
        gitUrl: 'https://github.com/tabbyml/tabby'
      } as Filepath

      const result = formatObjectToMarkdownBlock('file', unixGitObj, jsContent)
      const expectedMeta = JSON.stringify({ label: 'file', object: unixGitObj })

      expect(result).toContain('```context')
      expect(result).toContain(expectedMeta)
      expect(result).toContain(jsContent)
      expect(result).toContain('```')
    })

    it('should format Unix path with uri format', () => {
      const unixUriObj = {
        kind: 'uri',
        uri: '/home/user/projects/example.js'
      } as Filepath

      const result = formatObjectToMarkdownBlock('file', unixUriObj, jsContent)
      const expectedMeta = JSON.stringify({ label: 'file', object: unixUriObj })

      expect(result).toContain('```context')
      expect(result).toContain(expectedMeta)
      expect(result).toContain(jsContent)
      expect(result).toContain('```')
    })

    it('should format Windows path with uri format and backslashes', () => {
      const winUriObj = {
        kind: 'uri',
        uri: 'C:\\Users\\johndoe\\Projects\\example.js'
      } as Filepath

      const result = formatObjectToMarkdownBlock('file', winUriObj, jsContent)

      // Check for the structure, accounting for potential extra escaping by markdown stringifier
      const expectedLabelPart = '"label":"file"'
      const expectedObjectPart =
        '"object":{"kind":"uri","uri":"C:\\\\\\Users\\\\\\johndoe\\\\\\Projects\\\\\\example.js"}' // Match literal triple backslash

      expect(result).toContain('```context')
      expect(result).toContain(expectedLabelPart)
      expect(result).toContain(expectedObjectPart)
      expect(result).toContain(jsContent)
      expect(result).toContain('```')
    })
  })

  describe('markdown content handling', () => {
    const markdownContent = `# Example Markdown Document

This is a Markdown document with various elements.

## JavaScript Code Example

\`\`\`javascript
function hello() {
  console.log('Hello, world!');
}
\`\`\`

## Rust Code Example

\`\`\`rust
fn main() {
  println!("Hello from Rust!");
}
\`\`\`

## Blockquotes and Lists

> This is a blockquote

* List item 1
* List item 2
  * Nested list item
`

    it('should correctly handle markdown with nested code blocks', () => {
      const gitObj = {
        kind: 'git',
        filepath: '/home/user/docs/README.md',
        gitUrl: 'https://github.com/tabbyml/tabby'
      } as Filepath

      const result = formatObjectToMarkdownBlock(
        'file',
        gitObj,
        markdownContent
      )
      const expectedMeta = JSON.stringify({ label: 'file', object: gitObj })

      expect(result).toContain('```context')
      expect(result).toContain(expectedMeta)
      expect(result).toContain('```javascript')
      expect(result).toContain('```rust')
      expect(result).toContain('# Example Markdown Document')
      expect(result).toContain('```')
    })

    it('should correctly handle markdown with Windows paths', () => {
      const winUriObj = {
        kind: 'uri',
        uri: 'C:\\Users\\johndoe\\Documents\\README.md'
      } as Filepath

      const result = formatObjectToMarkdownBlock(
        'file',
        winUriObj,
        markdownContent
      )

      // Check for the structure, accounting for potential extra escaping by markdown stringifier
      const expectedLabelPart = '"label":"file"'
      const expectedObjectPart =
        '"object":{"kind":"uri","uri":"C:\\\\\\Users\\\\\\johndoe\\\\\\Documents\\\\\\README.md"}' // Match literal triple backslash

      expect(result).toContain('```context')
      expect(result).toContain(expectedLabelPart)
      expect(result).toContain(expectedObjectPart)
      expect(result).toContain('```javascript')
      expect(result).toContain('```rust')
      expect(result).toContain('```')
    })
  })

  describe('special cases', () => {
    it('should handle path with additional metadata', () => {
      const objWithMetadata = {
        kind: 'git',
        filepath: '/Users/johndoe/Developer/main.rs',
        gitUrl: 'https://github.com/tabbyml/tabby',
        line: 5,
        highlight: true
      } as Filepath

      const rustContent = `// Example Rust code
fn main() {
    println!("Hello, Rust!");
}`

      const result = formatObjectToMarkdownBlock(
        'file',
        objWithMetadata,
        rustContent
      )

      const expectedMeta = JSON.stringify({
        label: 'file',
        object: objWithMetadata
      })

      expect(result).toContain('```context')
      expect(result).toContain(expectedMeta)
      expect(result).toContain(rustContent)
      expect(result).toContain('```')
    })

    it('should handle special characters in paths', () => {
      const specialPathObj = {
        kind: 'git',
        filepath: '/Users/user/Projects/special-chars/file with spaces.js',
        gitUrl: 'https://github.com/tabbyml/tabby',
        branch: 'feature/new-branch'
      } as Filepath

      const jsContent = 'console.log("Special characters test");'

      const result = formatObjectToMarkdownBlock(
        'file',
        specialPathObj,
        jsContent
      )

      const expectedMeta = JSON.stringify({
        label: 'file',
        object: specialPathObj
      })

      expect(result).toContain('```context')
      expect(result).toContain(expectedMeta)
      expect(result).toContain(jsContent)
      expect(result).toContain('```')
    })

    it('should handle complex content types', () => {
      const winObj = {
        kind: 'uri',
        uri: 'D:\\Projects\\TypeScript\\interfaces.ts'
      } as Filepath

      const tsContent = `/**
 * User interface representing a person.
 */
interface User {
  id: number;
  name: string;
  email: string;
  isActive?: boolean; // Optional property
}

/**
 * Create a new user with default values.
 */
function createUser(name: string, email: string): User {
  return {
    id: Math.floor(Math.random() * 1000),
    name,
    email,
    isActive: true
  };
}

// Test the function
const newUser = createUser("John Doe", "john@example.com");
console.log(newUser);
`

      const result = formatObjectToMarkdownBlock('file', winObj, tsContent)

      // Check for the structure, accounting for potential extra escaping by markdown stringifier
      const expectedLabelPart = '"label":"file"'
      const expectedObjectPart =
        '"object":{"kind":"uri","uri":"D:\\\\\\Projects\\\\\\TypeScript\\\\\\interfaces.ts"}' // Match literal triple backslash

      expect(result).toContain('```context')
      expect(result).toContain(expectedLabelPart)
      expect(result).toContain(expectedObjectPart)
      expect(result).toContain(tsContent)
      expect(result).toContain('```')
    })
  })
})

describe('shouldAddPrefixNewline function', () => {
  it('should return false when index is at the start of text', () => {
    const result = shouldAddPrefixNewline(0, 'Some text here')
    expect(result).toBe(false)
  })

  it('should return false when there is a newline character before the index', () => {
    const result = shouldAddPrefixNewline(6, 'Hello\nworld')
    expect(result).toBe(false)
  })

  it('should return true when there is text before the index', () => {
    const result = shouldAddPrefixNewline(6, 'Hello world')
    expect(result).toBe(true)
  })

  it('should return false when there is only whitespace before the index', () => {
    const result = shouldAddPrefixNewline(3, '   Text')
    expect(result).toBe(false)
  })

  it('should handle mixed whitespace and text properly', () => {
    const result = shouldAddPrefixNewline(10, 'Hello    world')
    expect(result).toBe(true)
  })
})

describe('shouldAddSuffixNewline function', () => {
  it('should return false when index is at the end of text', () => {
    const text = 'Some text here'
    const result = shouldAddSuffixNewline(text.length, text)
    expect(result).toBe(false)
  })

  it('should return false when there is a newline character after the index', () => {
    const result = shouldAddSuffixNewline(5, 'Hello\nworld')
    expect(result).toBe(false)
  })

  it('should return true when there is text after the index', () => {
    const result = shouldAddSuffixNewline(5, 'Hello world')
    expect(result).toBe(true)
  })

  it('should return false when there is only whitespace after the index', () => {
    const result = shouldAddSuffixNewline(4, 'Text   ')
    expect(result).toBe(false)
  })

  it('should handle consecutive placeholder scenario correctly', () => {
    // Simulate two placeholders next to each other
    const text = '[[file:{}]][[file:{}]]'
    const firstPlaceholderEnd = 11

    const result = shouldAddSuffixNewline(firstPlaceholderEnd, text)
    expect(result).toBe(true)
  })
})

describe('formatObjectToMarkdownBlock with options', () => {
  it('should respect addPrefixNewline and addSuffixNewline options', () => {
    const unixGitObj = {
      kind: 'git',
      filepath: '/home/user/projects/example.js',
      gitUrl: 'https://github.com/tabbyml/tabby'
    } as Filepath

    const jsContent = 'console.log("Hello");'

    // With both newlines
    const resultBoth = formatObjectToMarkdownBlock(
      'file',
      unixGitObj,
      jsContent,
      {
        addPrefixNewline: true,
        addSuffixNewline: true
      }
    )
    expect(resultBoth.startsWith('\n')).toBe(true)
    expect(resultBoth.endsWith('\n')).toBe(true)

    // With no newlines
    const resultNone = formatObjectToMarkdownBlock(
      'file',
      unixGitObj,
      jsContent,
      {
        addPrefixNewline: false,
        addSuffixNewline: false
      }
    )
    expect(resultNone.startsWith('\n')).toBe(false)
    expect(resultNone.endsWith('\n')).toBe(false)

    // With only prefix newline
    const resultPrefix = formatObjectToMarkdownBlock(
      'file',
      unixGitObj,
      jsContent,
      {
        addPrefixNewline: true,
        addSuffixNewline: false
      }
    )
    expect(resultPrefix.startsWith('\n')).toBe(true)
    expect(resultPrefix.endsWith('\n')).toBe(false)

    // With only suffix newline
    const resultSuffix = formatObjectToMarkdownBlock(
      'file',
      unixGitObj,
      jsContent,
      {
        addPrefixNewline: false,
        addSuffixNewline: true
      }
    )
    expect(resultSuffix.startsWith('\n')).toBe(false)
    expect(resultSuffix.endsWith('\n')).toBe(true)
  })
})
