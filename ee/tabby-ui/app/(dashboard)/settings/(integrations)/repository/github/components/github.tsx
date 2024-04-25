'use client'

import Link from 'next/link'
import { useQuery } from 'urql'

import { ListGithubRepositoryProvidersQuery } from '@/lib/gql/generates/graphql'
import { listGithubRepositoryProviders } from '@/lib/tabby/query'
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconGitHub } from '@/components/ui/icons'
import LoadingWrapper from '@/components/loading-wrapper'

export default function GitProvidersPage() {
  const [{ data, fetching }] = useQuery({
    query: listGithubRepositoryProviders
  })
  const githubRepositoryProviders = data?.githubRepositoryProviders?.edges

  return (
    <>
      <LoadingWrapper loading={fetching}>
        {githubRepositoryProviders?.length ? (
          <div>
            <GitProvidersList data={githubRepositoryProviders} />
            <div className="mt-4 flex justify-end">
              <Link href="./github/new" className={buttonVariants()}>
                Create
              </Link>
            </div>
          </div>
        ) : (
          <GitProvidersPlaceholder />
        )}
      </LoadingWrapper>
    </>
  )
}

interface GitProvidersTableProps {
  data: ListGithubRepositoryProvidersQuery['githubRepositoryProviders']['edges']
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
                    <IconGitHub className="h-6 w-6" />
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
            <CardContent className="p-4 text-sm">
              <div className="flex border-b py-2">
                <span className="w-[30%] text-muted-foreground">Name</span>
                <span>{item.node.displayName}</span>
              </div>
              <div className="flex py-3">
                <span className="w-[30%] shrink-0 text-muted-foreground">
                  Status
                </span>
                <span>
                  {item.node?.connected
                    ? 'Connected'
                    : 'access_token needs update'}
                </span>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

const GitProvidersPlaceholder = () => {
  return (
    <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
      <div>No Data</div>
      <div className="flex justify-center">
        <Link
          href="./github/new"
          className={buttonVariants({ variant: 'default' })}
        >
          Create
        </Link>
      </div>
    </div>
  )
}
