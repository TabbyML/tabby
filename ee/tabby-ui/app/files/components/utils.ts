import { isNil } from 'lodash-es'

import fetcher from '@/lib/tabby/fetcher'
import { ResolveEntriesResponse, TFile } from '@/lib/types'

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

function getDirectoriesFromBasename(path: string, isDir?: boolean): string[] {
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

async function fetchEntriesFromPath(path: string | undefined) {
  if (!path) return []
  const repoName = resolveRepoNameFromPath(path)
  const basename = resolveBasenameFromPath(path)
  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(basename)
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(`/repositories/${repoName}/resolve/${dir}`).catch(e => [])
    )
  const entries = await Promise.all(requests.map(fn => fn()))
  let result: TFile[] = []
  for (let entry of entries) {
    if (entry?.entries?.length) {
      result = [...result, ...entry.entries]
    }
  }
  return result
}

export {
  resolveRepoNameFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  fetchEntriesFromPath
}
