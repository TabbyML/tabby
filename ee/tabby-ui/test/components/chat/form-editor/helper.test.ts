import { describe, expect, test } from 'vitest'

import {
  extractRepoNameFromGitUrl,
  formatFileDescription
} from '../../../../components/chat/form-editor/helper'

describe('extractRepoNameFromGitUrl', () => {
  test('should extract repo name from GitHub HTTPS URL', () => {
    const url = 'https://github.com/TabbyML/tabby.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('tabby')
  })

  test('should extract repo name from GitLab HTTPS URL', () => {
    const url = 'https://gitlab.com/organization/project-name.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('project-name')
  })

  test('should extract repo name from BitBucket HTTPS URL', () => {
    const url = 'https://bitbucket.org/user/repo-name.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('repo-name')
  })

  test('should extract repo name from GitHub SSH URL', () => {
    const url = 'git@github.com:TabbyML/tabby.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('tabby')
  })

  test('should extract repo name from GitLab SSH URL', () => {
    const url = 'git@gitlab.com:organization/project-name.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('project-name')
  })

  test('should extract repo name from BitBucket SSH URL', () => {
    const url = 'git@bitbucket.org:user/repo-name.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('repo-name')
  })

  test('should extract repo name from Git protocol URL', () => {
    const url = 'git://github.com/TabbyML/tabby.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('tabby')
  })
  test('should handle URLs without .git suffix', () => {
    const httpsUrl = 'https://github.com/TabbyML/tabby'
    const sshUrl = 'git@github.com:TabbyML/tabby'

    expect(extractRepoNameFromGitUrl(httpsUrl)).toBe('tabby')
    expect(extractRepoNameFromGitUrl(sshUrl)).toBe('tabby')
  })

  test('should handle complex repository names', () => {
    const url1 = 'https://github.com/organization/repo-with-dashes.git'
    const url2 = 'git@github.com:organization/repo.with.dots.git'
    const url3 = 'https://github.com/organization/repo_with_underscores.git'

    expect(extractRepoNameFromGitUrl(url1)).toBe('repo-with-dashes')
    expect(extractRepoNameFromGitUrl(url2)).toBe('repo.with.dots')
    expect(extractRepoNameFromGitUrl(url3)).toBe('repo_with_underscores')
  })

  test('should handle self-hosted Git services', () => {
    const url1 = 'https://git.company.com/team/project.git'
    const url2 = 'git@git.company.com:team/project.git'

    expect(extractRepoNameFromGitUrl(url1)).toBe('project')
    expect(extractRepoNameFromGitUrl(url2)).toBe('project')
  })

  test('should handle URLs with port numbers', () => {
    const url = 'https://git.company.com:8080/team/project.git'
    expect(extractRepoNameFromGitUrl(url)).toBe('project')
  })

  test('should handle edge cases', () => {
    expect(extractRepoNameFromGitUrl('')).toBe('')

    expect(extractRepoNameFromGitUrl(undefined as any)).toBe('')

    expect(extractRepoNameFromGitUrl('not-a-url')).toBe('')

    expect(extractRepoNameFromGitUrl('https://github.com/')).toBe('')
    expect(extractRepoNameFromGitUrl('git@github.com:')).toBe('')

    expect(
      extractRepoNameFromGitUrl('https://github.com/user/repo.git?param=value')
    ).toBe('')
  })
})

describe('formatFileDescription', () => {
  test('should format description for file with GitHub gitUrl', () => {
    const fileData = {
      filepath: 'src/main.ts',
      gitUrl: 'https://github.com/TabbyML/tabby.git'
    }
    expect(formatFileDescription(fileData)).toBe('tabby - src/main.ts')
  })

  test('should format description for file with GitLab gitUrl', () => {
    const fileData = {
      filepath: 'lib/utils/index.js',
      gitUrl: 'https://gitlab.com/organization/project.git'
    }
    expect(formatFileDescription(fileData)).toBe('project - lib/utils/index.js')
  })

  test('should format description for file with SSH gitUrl', () => {
    const fileData = {
      filepath: 'README.md',
      gitUrl: 'git@github.com:TabbyML/tabby.git'
    }
    expect(formatFileDescription(fileData)).toBe('tabby - README.md')
  })

  test('should format description for file with commit', () => {
    const fileData = {
      filepath: 'src/index.ts',
      gitUrl: 'https://github.com/TabbyML/tabby.git',
      commit: 'abc123'
    }
    expect(formatFileDescription(fileData)).toBe('tabby - src/index.ts')
  })

  test('should return filepath when no gitUrl is present', () => {
    const fileData = {
      filepath: 'src/components/App.tsx'
    }
    expect(formatFileDescription(fileData)).toBe('src/components/App.tsx')
  })

  test('should handle invalid gitUrl', () => {
    const fileData = {
      filepath: 'package.json',
      gitUrl: 'invalid-url'
    }
    expect(formatFileDescription(fileData)).toBe('package.json')
  })

  test('should handle file with baseDir', () => {
    const fileData = {
      filepath: 'src/utils.ts',
      baseDir: '/project'
    }
    expect(formatFileDescription(fileData)).toBe('project - src/utils.ts')
  })

  test('should prioritize gitUrl over baseDir', () => {
    const fileData = {
      filepath: 'config.json',
      gitUrl: 'https://github.com/TabbyML/tabby.git',
      baseDir: '/some/local/path'
    }
    expect(formatFileDescription(fileData)).toBe('tabby - config.json')
  })

  test('should handle edge cases', () => {
    expect(formatFileDescription(null as any)).toBe('')

    expect(formatFileDescription({} as any)).toBe('')

    expect(formatFileDescription({ filepath: '' })).toBe('')

    const fileData = {
      filepath: 'file.txt',
      gitUrl: 'https://example.com/'
    }
    expect(formatFileDescription(fileData)).toBe('file.txt')
  })
})
