'use client'

import Link from 'next/link'
import { useQuery } from 'urql'

import { ListGithubRepositoryProvidersQuery } from '@/lib/gql/generates/graphql'
import { listGithubRepositoryProviders } from '@/lib/tabby/query'
import { Button, buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconGitHub } from '@/components/ui/icons'
import LoadingWrapper from '@/components/loading-wrapper'

import { RepositoryHeader } from './header'
import fetcher from '@/lib/tabby/fetcher'
import { getAuthToken } from '@/lib/tabby/token-management'

export default function GitProvidersPage() {
  const [{ data }] = useQuery({ query: listGithubRepositoryProviders })
  const githubRepositoryProviders = data?.githubRepositoryProviders?.edges

  return (
    <>
      <RepositoryHeader />
      <LoadingWrapper loading={false}>
        {githubRepositoryProviders?.length ? (
          <div>
            <GitProvidersList data={githubRepositoryProviders} />
            <div className="mt-4 flex justify-end">
              <Link href="/settings/gitops/new" className={buttonVariants()}>
                Add A Git Provider
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

  const handleConnect = async (id: string) => {
    const width = 600, height = 600;
    const left = (window.innerWidth - width) / 2;
    const top = (window.innerHeight - height) / 2;

    await fetch(`/integrations/github/connect/${id}`, {
      headers: {
        'Authorization': `Bearer ${getAuthToken()?.accessToken}`,
      },
      redirect: 'manual'
    }).then(response => {
      // 处理响应
      if (response.type === 'opaqueredirect') {
        // 获取重定向的 URL
        const redirectUrl = response.url;
        debugger
        // 根据重定向 URL 打开小窗口进行 OAuth 认证
        if (redirectUrl) {
          window.open(redirectUrl, 'GitHubAuth', `toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, copyhistory=no, width=${width}, height=${height}, top=${top}, left=${left}`);
        } else {
          throw new Error('Redirect URL not found');
        }
      } else {
        // 处理非重定向响应
        console.log('Not a redirect response');
      }
    })

    // window.open(url, 'GitHubAuth', `toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, copyhistory=no, width=${width}, height=${height}, top=${top}, left=${left}`);
  }


  return (
    <div className="space-y-8">
      {data?.map(item => {
        return (
          <Card key={item.node.id}>
            <CardHeader className="border-b p-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">
                  <div className="flex items-center gap-2">
                    <IconGitHub className="h-8 w-8" />
                    GitHub.com
                  </div>
                </CardTitle>
                <Button onClick={e => handleConnect(item.node.id)}>connect</Button>
                <Link
                  href={`/settings/gitops/detail?id=${item.node.id}`}
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
              <div className="flex py-3 border-b">
                <span className="w-[30%] text-muted-foreground shrink-0">
                  Application ID
                </span>
                <span className="truncate">{item.node.applicationId}</span>
              </div>
              <div className="flex py-3">
                <span className="w-[30%] text-muted-foreground shrink-0">
                  Linked repositories
                </span>
                {/* todo: add query */}
                <span className="truncate">2</span>
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
          href="/settings/gitops/new"
          className={buttonVariants({ variant: 'default' })}
        >
          Add A Git Provider
        </Link>
      </div>
    </div>
  )
}
