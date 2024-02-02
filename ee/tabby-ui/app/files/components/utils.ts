import { isNil } from 'lodash-es'

function resolveRepoNameFromPath(path: string | undefined) {
  if (!path) return ''
  return path.split('/')?.[0]
}

function resolveBasenameFromPath(path?: string) {
  if (!path) return ''
  return path.split('/').slice(1).join('/')
}

function resolveFileNameFromPath(path: string) {
  if (!path) return ''
  const pathSegments = path.split('/')
  return pathSegments[pathSegments.length - 1]
}

function getDirectoriesFromPath(path: string, isDir?: boolean): string[] {
  if (isNil(path)) return []

  let result = ['']
  const pathSegments = path.split('/')
  // if path points to a file, the dirs do not include the path itself
  const len = isDir ? pathSegments.length : pathSegments.length - 1
  for (let i = 0; i < len; i++) {
    result.push(pathSegments.slice(0, i + 1).join('/'))
  }
  return result
}

export {
  resolveRepoNameFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  getDirectoriesFromPath
}
