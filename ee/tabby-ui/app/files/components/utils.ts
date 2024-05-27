import { isNil, keyBy } from 'lodash-es'

import {
  RepositoryKind,
  RepositoryListQuery
} from '@/lib/gql/generates/graphql'
import fetcher from '@/lib/tabby/fetcher'
import { ResolveEntriesResponse, TFile } from '@/lib/types'

function resolveRepositoryInfoFromPath(path: string | undefined): {
  repositoryKind?: RepositoryKind
  repositoryName?: string
  basename?: string
  repositorySpecifier?: string
} {
  const emptyResult = {}
  if (!path) return emptyResult
  const pathSegments = path.split('/')
  const repositoryKindStr = pathSegments[0]

  if (!repositoryKindStr) {
    return emptyResult
  }

  if (repositoryKindStr === 'git') {
    if (pathSegments.length < 2) return emptyResult

    const repositoryName = pathSegments[1]
    return {
      repositoryKind: RepositoryKind.Git,
      repositoryName,
      basename: pathSegments.slice(2).join('/'),
      repositorySpecifier: `git/${repositoryName}`
    }
  } else if (['github', 'gitlab'].includes(repositoryKindStr)) {
    if (pathSegments.length < 3) return emptyResult
    const kind =
      repositoryKindStr === 'github'
        ? RepositoryKind.Github
        : RepositoryKind.Gitlab
    const repositoryName = [pathSegments[1], pathSegments[2]].join('/')

    return {
      repositoryKind: kind,
      repositoryName,
      basename: pathSegments.slice(3).join('/'),
      repositorySpecifier: `${kind.toLowerCase()}/${repositoryName}`
    }
  }
  return emptyResult
}

function resolveFileNameFromPath(path: string) {
  if (!path) return ''
  const pathSegments = path.split('/')
  return pathSegments[pathSegments.length - 1]
}

function getDirectoriesFromBasename(
  path: string | undefined,
  isDir?: boolean
): string[] {
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

async function fetchEntriesFromPath(
  path: string | undefined,
  repository: RepositoryListQuery['repositoryList'][0] | undefined
) {
  if (!path || !repository) return []

  const { basename } = resolveRepositoryInfoFromPath(path)
  // array of dir basename that do not include the repo name.
  const directoryPaths = getDirectoriesFromBasename(basename)
  // fetch all dirs from path
  const requests: Array<() => Promise<ResolveEntriesResponse>> =
    directoryPaths.map(
      dir => () =>
        fetcher(
          encodeURIComponentIgnoringSlash(
            `/repositories/${repository.kind.toLowerCase()}/${
              repository.id
            }/resolve/${dir}`
          )
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
  repo:
    | { kind: RepositoryKind | undefined; name: string | undefined }
    | undefined
) {
  if (repo?.kind && repo?.name) {
    return `${repo.kind.toLowerCase()}/${repo.name}`
  }

  return undefined
}

function repositoryList2Map(repos: RepositoryListQuery['repositoryList']) {
  return keyBy(repos, o => `${o.kind.toLowerCase()}/${o.name}`)
}

function encodeURIComponentIgnoringSlash(str: string) {
  return str
    .split('/')
    .map(part => encodeURIComponent(part))
    .join('/')
}

export {
  resolveRepoSpecifierFromRepoInfo,
  resolveFileNameFromPath,
  getDirectoriesFromBasename,
  fetchEntriesFromPath,
  resolveRepositoryInfoFromPath,
  repositoryList2Map,
  encodeURIComponentIgnoringSlash
}
