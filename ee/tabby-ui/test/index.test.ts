import { describe, expect, it } from 'vitest'
import { findClosestGitRepository, canonicalizeUrl } from '../lib/utils/repository'
import type { GitRepository } from 'tabby-chat-panel'

describe('canonicalizeUrl', () => {
  it('should remove auth info from URL', () => {
    const result = canonicalizeUrl('https://abc:dev@github.com/');
    expect(result).toBe('https://github.com/');
  });

  it('should remove token from URL', () => {
    const result = canonicalizeUrl('https://token@github.com/TabbyML/tabby');
    expect(result).toBe('https://github.com/TabbyML/tabby');
  });

  it('should return the same URL if no auth info is present', () => {
    const result = canonicalizeUrl('https://github.com/TabbyML/tabby');
    expect(result).toBe('https://github.com/TabbyML/tabby');
  });

  it('should remove .git suffix from URL', () => {
    const result = canonicalizeUrl('https://github.com/TabbyML/tabby.git');
    expect(result).toBe('https://github.com/TabbyML/tabby');
  });

  it('should handle file URLs correctly', () => {
    const result = canonicalizeUrl('file:///home/TabbyML/tabby');
    expect(result).toBe('file:///home/TabbyML/tabby');
  });
});

describe('findClosestGitRepository', () => {
  it('should match .git suffix', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/example/test' },
    ]
    const result = findClosestGitRepository(repositories, 'https://github.com/example/test.git')
    expect(result).toEqual(repositories[0])
  })

  it('should match auth in URL', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/example/test' },
    ]
    const result = findClosestGitRepository(repositories, 'https://creds@github.com/example/test')
    expect(result).toEqual(repositories[0])
  })

  it('should not match different names', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/example/anoth-repo' },
    ]
    const result = findClosestGitRepository(repositories, 'https://github.com/example/another-repo')
    expect(result).toBeUndefined()
  })

  it('should not match repositories with a common prefix', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/registry-tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'https://github.com/TabbyML/tabby')
    expect(result).toBeUndefined()
  })

  it('should not match entirely different repository names', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/uptime' },
    ]
    const result = findClosestGitRepository(repositories, 'https://github.com/TabbyML/tabby')
    expect(result).toBeUndefined()
  })

  it('should not match URL without repository name', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'https://github.com')
    expect(result).toBeUndefined()
  })

  it('should not match different host', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'https://bitbucket.com/TabbyML/tabby')
    expect(result).toBeUndefined()
  })

  it('should not match multiple close matches', () => {
    const repositories: GitRepository[] = [
      { url: 'https://bitbucket.com/CrabbyML/crabby' },
      { url: 'https://gitlab.com/TabbyML/registry-tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'git@github.com:TabbyML/tabby')
    expect(result).toBeUndefined()
  })

  it('should match different protocol and suffix', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'git@github.com:TabbyML/tabby.git')
    expect(result).toEqual(repositories[0])
  })

  it('should match different protocol', () => {
    const repositories: GitRepository[] = [
      { url: 'https://github.com/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'git@github.com:TabbyML/tabby')
    expect(result).toEqual(repositories[0])
  })

  it('should match URL without organization', () => {
    const repositories: GitRepository[] = [
      { url: 'https://custom-git.com/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'https://custom-git.com/tabby')
    expect(result).toEqual(repositories[0])
  })

  it('should match local URL', () => {
    const repositories: GitRepository[] = [
      { url: 'file:///home/TabbyML/tabby' },
    ]
    const result = findClosestGitRepository(repositories, 'git@github.com:TabbyML/tabby.git')
    expect(result).toEqual(repositories[0])
  })
})
