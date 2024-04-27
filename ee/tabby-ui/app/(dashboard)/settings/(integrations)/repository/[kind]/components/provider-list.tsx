'use client'

import Link from 'next/link'
import { useQuery } from 'urql'

import {
  RepositoryKind,
  RepositoryProviderStatus
} from '@/lib/gql/generates/graphql'
import {
  listGithubRepositoryProviders,
  listGitlabRepositoryProviders
} from '@/lib/tabby/query'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import LoadingWrapper from '@/components/loading-wrapper'

import { useRepositoryKind } from '../hooks/use-repository-kind'

interface GitProvidersListProps {
  kind: RepositoryKind
}

export default function RepositoryProvidersPage({
  kind
}: GitProvidersListProps) {
  if (kind === RepositoryKind.Github) {
    return <GithubProviders />
  }

  if (kind === RepositoryKind.Gitlab) {
    return <GitlabProviders />
  }

  return <div>404</div>
}

function GithubProviders() {
  const [{ data, fetching }] = useQuery({
    query: listGithubRepositoryProviders
  })
  const providers = data?.githubRepositoryProviders?.edges

  return <RepositoryProvidersView fetching={fetching} providers={providers} />
}

function GitlabProviders() {
  const [{ data, fetching }] = useQuery({
    query: listGitlabRepositoryProviders
  })
  const providers = data?.gitlabRepositoryProviders?.edges

  return <RepositoryProvidersView fetching={fetching} providers={providers} />
}

interface RepositoryProvidersViewProps {
  fetching: boolean
  providers:
    | Array<{
        node: {
          id: string
          displayName: string
          status: RepositoryProviderStatus
        }
      }>
    | undefined
}
function RepositoryProvidersView({
  fetching,
  providers
}: RepositoryProvidersViewProps) {
  return (
    <LoadingWrapper loading={fetching}>
      {providers?.length ? (
        <>
          <GitProvidersList data={providers} />
          <CreateRepositoryProvider />·
        </>
      ) : (
        <GitProvidersPlaceholder />
      )}
    </LoadingWrapper>
  )
}

interface GitProvidersTableProps {
  data: RepositoryProvidersViewProps['providers']
}
const GitProvidersList: React.FC<GitProvidersTableProps> = ({ data }) => {
  return (
    <div className="space-y-8">
      {data?.map(item => {
        return (
          <Card key={item.node.id}>
            <CardHeader className="border-b p-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">
                  <div className="flex items-center gap-2">
                    {item.node.displayName}
                  </div>
                </CardTitle>
                <Link
                  href={`github/detail?id=${item.node.id}`}
                  className={buttonVariants({ variant: 'secondary' })}
                >
                  View
                </Link>
              </div>
            </CardHeader>
            <CardContent className="p-0 text-sm">
              <div className="flex px-8 py-4">
                <span className="w-[30%] shrink-0 text-muted-foreground">
                  Status
                </span>
                <span>{toStatusMessage(item.node.status)}</span>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

const CreateRepositoryProvider = () => {
  const kind = useRepositoryKind()
  return (
    <div className="mt-4 flex justify-end">
      <Link
        href={`./${kind.toLocaleLowerCase()}/new`}
        className={buttonVariants()}
      >
        Create
      </Link>
    </div>
  )
}

function toStatusMessage(status: RepositoryProviderStatus) {
  switch (status) {
    case RepositoryProviderStatus.Ready:
      return 'Ready'
    case RepositoryProviderStatus.Error:
      return 'Processing error. Please check if the access token is still valid'
    case RepositoryProviderStatus.Pending:
      return 'Awaiting the next data synchronization'
  }
}

const GitProvidersPlaceholder = () => {
  const kind = useRepositoryKind()
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
      <div>No Data</div>
      <div className="flex justify-center">
        <Link
          href={`./${kind.toLocaleLowerCase()}/new`}
          className={buttonVariants({ variant: 'default' })}
        >
          Create
        </Link>
      </div>
    </div>
  )
}
