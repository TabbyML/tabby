'use client'

import React from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'
import { useQuery } from 'urql'

import {
  IntegrationStatus,
  ListIntegrationsQuery
} from '@/lib/gql/generates/graphql'
import { listIntegrations } from '@/lib/tabby/query'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import LoadingWrapper from '@/components/loading-wrapper'

import { useIntegrationKind } from '../hooks/use-repository-kind'

export default function RepositoryProvidersPage() {
  return <ProviderList />
}

function ProviderList() {
  const kind = useIntegrationKind()
  const [{ data, fetching }] = useQuery({
    query: listIntegrations,
    variables: { kind }
  })

  const providers = data?.integrations?.edges

  return <RepositoryProvidersView fetching={fetching} providers={providers} />
}

interface RepositoryProvidersViewProps {
  fetching: boolean
  providers: ListIntegrationsQuery['integrations']['edges'] | undefined
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

function toStatusMessage(status: IntegrationStatus) {
  switch (status) {
    case IntegrationStatus.Ready:
      return 'Ready'
    case IntegrationStatus.Failed:
      return 'Processing error. Please check if the access token is still valid'
    case IntegrationStatus.Pending:
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
