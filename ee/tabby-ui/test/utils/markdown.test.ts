import { describe, expect, it } from 'vitest'
import { formatObjectToMarkdownBlock } from '../../lib/utils/markdown'
import { Filepath } from 'tabby-chat-panel/index';

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
`;

    it('should format Unix path with git format', () => {
      const unixGitObj = { 
        kind: "git", 
        filepath: '/home/user/projects/example.js', 
        gitUrl: "https://github.com/tabbyml/tabby" 
      } as Filepath;
      
      const result = formatObjectToMarkdownBlock('file', unixGitObj, jsContent);
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${JSON.stringify(unixGitObj)}`);
      expect(result).toContain(jsContent);
      expect(result).toContain('```');
    });

    it('should format Unix path with uri format', () => {
      const unixUriObj = { 
        kind: "uri", 
        uri: '/home/user/projects/example.js' 
      } as Filepath;
      
      const result = formatObjectToMarkdownBlock('file', unixUriObj, jsContent);
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${JSON.stringify(unixUriObj)}`);
      expect(result).toContain(jsContent);
      expect(result).toContain('```');
    });

    it('should format Windows path with uri format and backslashes', () => {
      const winUriObj = { 
        kind: "uri", 
        uri: 'C:\\Users\\johndoe\\Projects\\example.js' 
      } as Filepath;
      
      const result = formatObjectToMarkdownBlock('file', winUriObj, jsContent);
      
      const expectedJson = JSON.stringify(winUriObj).replace(/\\\\/g, '\\\\\\');
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${expectedJson}`);
      expect(result).toContain(jsContent);
      expect(result).toContain('```');
    });
  });

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
`;

    it('should correctly handle markdown with nested code blocks', () => {
      const gitObj = { 
        kind: "git", 
        filepath: '/home/user/docs/README.md', 
        gitUrl: "https://github.com/tabbyml/tabby" 
      } as Filepath;
      
      const result = formatObjectToMarkdownBlock('file', gitObj, markdownContent);
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${JSON.stringify(gitObj)}`);
      expect(result).toContain('```javascript');
      expect(result).toContain('```rust');
      expect(result).toContain('# Example Markdown Document');
      expect(result).toContain('```');
    });

    it('should correctly handle markdown with Windows paths', () => {
      const winUriObj = { 
        kind: "uri", 
        uri: 'C:\\Users\\johndoe\\Documents\\README.md' 
      } as Filepath;
      
      const result = formatObjectToMarkdownBlock('file', winUriObj, markdownContent);
      
      const expectedJson = JSON.stringify(winUriObj).replace(/\\\\/g, '\\\\\\');
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${expectedJson}`);
      expect(result).toContain('```javascript');
      expect(result).toContain('```rust');
      expect(result).toContain('```');
    });
  });

  describe('special cases', () => {
    it('should handle path with additional metadata', () => {
      const objWithMetadata = { 
        kind: "git",
        filepath: '/Users/johndoe/Developer/main.rs',
        gitUrl: "https://github.com/tabbyml/tabby",
        line: 5,
        highlight: true
      } as Filepath;
      
      const rustContent = `// Example Rust code
fn main() {
    println!("Hello, Rust!");
}`;
      
      const result = formatObjectToMarkdownBlock('file', objWithMetadata, rustContent);
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${JSON.stringify(objWithMetadata)}`);
      expect(result).toContain(rustContent);
      expect(result).toContain('```');
    });

    it('should handle special characters in paths', () => {
      const specialPathObj = { 
        kind: "git",
        filepath: '/Users/user/Projects/special-chars/file with spaces.js',
        gitUrl: "https://github.com/tabbyml/tabby",
        branch: 'feature/new-branch'
      } as Filepath;
      
      const jsContent = 'console.log("Special characters test");';
      
      const result = formatObjectToMarkdownBlock('file', specialPathObj, jsContent);
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${JSON.stringify(specialPathObj)}`);
      expect(result).toContain(jsContent);
      expect(result).toContain('```');
    });

    it('should handle complex content types', () => {
      const winObj = { 
        kind: "uri", 
        uri: 'D:\\Projects\\TypeScript\\interfaces.ts' 
      } as Filepath;
      
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
`;
      
      const result = formatObjectToMarkdownBlock('file', winObj, tsContent);
      
      const expectedJson = JSON.stringify(winObj).replace(/\\\\/g, '\\\\\\');
      
      expect(result).toContain('```context label=file');
      expect(result).toContain(`object=${expectedJson}`);
      expect(result).toContain(tsContent);
      expect(result).toContain('```');
    });
  });
});