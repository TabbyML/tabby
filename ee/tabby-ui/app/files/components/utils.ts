import { isNil } from 'lodash-es'

import { RepositoryKind } from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { ResolveEntriesResponse, TFile } from '@/lib/types'

function resolveRepoKindFromPath(path: string | undefined) {
  if (!path) return ''
  return path.split('/')?.[0]
}

function resolveRepoIdFromPath(path: string | undefined) {
  if (!path) return ''
  return path.split('/')?.[1]
}

function resolveBasenameFromPath(path?: string) {
  if (!path) return ''
  return path.split('/').slice(2).join('/')
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
  const repoId = resolveRepoIdFromPath(path)
  const repoKind = resolveRepoKindFromPath(path)
  if (!path || !repoId || !repoKind) return []

  const basename = resolveBasenameFromPath(path)
  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(basename)
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(
          `/repositories/${repoKind.toLowerCase()}/${repoId}/resolve/${dir}`
        ).catch(e => [])
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

function resolveRepoSpecifierFromRepoInfo(
  repo: { kind: RepositoryKind | string; id: string } | undefined
) {
  if (repo?.kind && repo?.id) {
    return `${repo.kind.toLowerCase()}/${repo.id}`
  }

  return undefined
}

function resolveRepoSpecifierFromPath(path: string | undefined) {
  if (!path) return ''
  let pathSegments = path.split('/')
  if (pathSegments.length < 2) return ''
  return [pathSegments[0], pathSegments[1]].join('/')
}

function key2RepositoryKind(key: string) {
  const map: Record<string, RepositoryKind> = {
    git: RepositoryKind.Git,
    github: RepositoryKind.Github,
    gitlab: RepositoryKind.Gitlab
  }
  return map[key] || undefined
}

export {
  resolveRepoKindFromPath,
  resolveRepoIdFromPath,
  resolveRepoSpecifierFromRepoInfo,
  resolveRepoSpecifierFromPath,
  resolveBasenameFromPath,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  key2RepositoryKind,
  fetchEntriesFromPath
}
