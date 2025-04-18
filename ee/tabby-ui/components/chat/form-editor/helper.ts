/**
 * Extracts repository name from different Git URL formats
 * Supports HTTPS, SSH, and Git protocol URLs
 *
 * @param gitUrl The Git URL to parse
 * @returns The extracted repository name, or empty string if extraction fails
 */
export function extractRepoNameFromGitUrl(gitUrl: string): string {
  if (!gitUrl) return ''

  try {
    if (gitUrl.includes('@') && gitUrl.includes(':')) {
      const parts = gitUrl.split(':')
      if (parts.length > 1) {
        const repoPath = parts[1].split('/')
        const repoName = repoPath[repoPath.length - 1] || ''
        return repoName.replace(/\.git$/, '')
      }
    } else {
      try {
        if (gitUrl.includes('?')) {
          return ''
        }

        const url = new URL(gitUrl)
        const pathParts = url.pathname.split('/').filter(Boolean)
        if (pathParts.length > 0) {
          const repoName = pathParts[pathParts.length - 1]
          return repoName.replace(/\.git$/, '')
        }
      } catch {
        return ''
      }
    }

    return ''
  } catch (e) {
    return ''
  }
}

/**
 * Formats file information for display in mention list
 *
 * @param fileData The file data from convertFromFilepath
 * @returns A formatted description string for display
 */
export function formatFileDescription(fileData: {
  filepath: string
  gitUrl?: string
  baseDir?: string
  commit?: string
}): string {
  if (!fileData) return ''
  if (!fileData.filepath) return ''

  // If this file has a gitUrl, show repo name and relative path
  if (fileData.gitUrl) {
    const repoName = extractRepoNameFromGitUrl(fileData.gitUrl)

    if (repoName) {
      return `${repoName} - ${fileData.filepath}`
    }
  }

  // If this is a workspace file with baseDir, show the baseDir
  if (fileData.baseDir) {
    const baseDirName = fileData.baseDir.split('/').pop() || fileData.baseDir
    return `${baseDirName} - ${fileData.filepath}`
  }

  // For other files, just return the filepath
  return fileData.filepath
}
