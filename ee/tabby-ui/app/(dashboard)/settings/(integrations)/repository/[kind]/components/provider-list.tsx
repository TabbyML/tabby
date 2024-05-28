'use client'

import Link from 'next/link'
import { useQuery } from 'urql'

import {
  RepositoryKind,
  RepositoryProviderStatus
} from '@/lib/gql/generates/graphql'
import {
  listGithubRepositoryProviders,
  listGithubSelfHostedRepositoryProviders,
  listGitlabRepositoryProviders,
  listGitlabSelfHostedRepositoryProviders
} from '@/lib/tabby/query'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import LoadingWrapper from '@/components/loading-wrapper'

import React from 'react'
import { useParams } from 'next/navigation'
import { QueryResponseData } from '@/lib/tabby/gql'

interface GitProvidersListProps {
  kind: RepositoryKind
}

type TProviderList =
  | Array<{
      node: {
        id: string
        displayName: string
        status: RepositoryProviderStatus
        apiBase?: string
      }
    }>
  | undefined

export default function RepositoryProvidersPage({
  kind
}: GitProvidersListProps) {
  return <ProviderList kind={kind} key={kind} />
  // return <div>404</div>
}

function ProviderList({ kind }: GitProvidersListProps) {
  const query = React.useMemo(() => {
    switch (kind) {
      case RepositoryKind.Github:
        return listGithubRepositoryProviders
      case RepositoryKind.GithubSelfHosted:
        return listGithubSelfHostedRepositoryProviders
      case RepositoryKind.Gitlab:
        return listGitlabRepositoryProviders
      case RepositoryKind.GitlabSelfHosted:
        return listGitlabSelfHostedRepositoryProviders
    }
  }, [kind])

  const resolver = React.useMemo(() => {
    // todo also return pageInfo for pagination
    switch (kind) {
      case RepositoryKind.Github:
        return (res: QueryResponseData<typeof listGithubRepositoryProviders>) =>
          res?.githubRepositoryProviders?.edges
      case RepositoryKind.GithubSelfHosted:
        return (
          res: QueryResponseData<typeof listGithubSelfHostedRepositoryProviders>
        ) => res?.githubSelfHostedRepositoryProviders?.edges
      case RepositoryKind.Gitlab:
        return (res: QueryResponseData<typeof listGitlabRepositoryProviders>) =>
          res?.gitlabRepositoryProviders?.edges
      case RepositoryKind.GitlabSelfHosted:
        return (
          res: QueryResponseData<typeof listGitlabSelfHostedRepositoryProviders>
        ) => res?.gitlabSelfHostedRepositoryProviders?.edges
      default:
        return () => []
    }
  }, [kind]) as (response: any) => TProviderList

  const [{ data, fetching }] = useQuery({
    query: query as any,
    pause: !query
  })

  const providers = resolver(data)

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
          apiBase?: string
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
          <CreateRepositoryProvider />
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
  const params = useParams()
  return (
    <div className="space-y-8">
      {data?.map(item => {
        return (
          <Card key={item.node.id}>
            <CardHeader className="border-b px-6 py-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">
                  <div className="flex items-center gap-2">
                    {item.node.displayName}
                  </div>
                </CardTitle>
                <Link
                  href={`${params.kind}/detail?id=${item.node.id}`}
                  className={buttonVariants({ variant: 'secondary' })}
                >
                  View
                </Link>
              </div>
            </CardHeader>
            <CardContent className="p-0 text-sm">
              <div className="flex px-6 py-4">
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
  const params = useParams()
  return (
    <div className="mt-4 flex justify-end">
      <Link href={`./${params.kind}/new`} className={buttonVariants()}>
        Create
      </Link>
    </div>
  )
}

function toStatusMessage(status: RepositoryProviderStatus) {
  switch (status) {
    case RepositoryProviderStatus.Ready:
      return 'Ready'
    case RepositoryProviderStatus.Failed:
      return 'Processing error. Please check if the access token is still valid'
    case RepositoryProviderStatus.Pending:
      return 'Awaiting the next data synchronization'
  }
}

const GitProvidersPlaceholder = () => {
  const params = useParams()
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
      <div>No Data</div>
      <div className="flex justify-center">
        <Link
          href={`./${params.kind}/new`}
          className={buttonVariants({ variant: 'default' })}
        >
          Create
        </Link>
      </div>
    </div>
  )
}
