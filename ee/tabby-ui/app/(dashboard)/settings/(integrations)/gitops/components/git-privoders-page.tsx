'use client'

import Link from 'next/link'
import useLocalStorage from 'use-local-storage'

import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconGitHub } from '@/components/ui/icons'
import LoadingWrapper from '@/components/loading-wrapper'

import { BasicInfoFormValues } from './basic-info-form'
import { RepositoryHeader } from './header'
import { OAuthApplicationFormValues } from './oauth-application-form'

type ProviderItem = BasicInfoFormValues &
  OAuthApplicationFormValues & { id: number }

export default function GitProvidersPage() {
  // todo remove
  const [mockGitopsData, setMockGitopsData] =
    useLocalStorage<Array<ProviderItem> | null>('mock-gitops-data', null)

  return (
    <>
      <RepositoryHeader />
      <LoadingWrapper loading={false}>
        {mockGitopsData?.length ? (
          <div>
            <GitProvidersList data={mockGitopsData} />
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
  data: Array<ProviderItem>
}

const GitProvidersList: React.FC<GitProvidersTableProps> = ({ data }) => {
  return (
    <div className="space-y-8">
      {data?.map(item => {
        return (
          <Card key={item.id}>
            <CardHeader className="border-b p-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-xl">
                  <div className="flex items-center gap-2">
                    <IconGitHub className="w-8 h-8" />
                    GitHub.com
                  </div>
                </CardTitle>
                <Link
                  href={`/settings/gitops/detail?id=${item.id}`}
                  className={buttonVariants({ variant: 'secondary' })}
                >
                  View
                </Link>
              </div>
            </CardHeader>
            <CardContent className="p-4 text-sm">
              <div className="flex border-b py-2">
                <span className="w-[30%] text-muted-foreground font-semibold">
                  Instance URL
                </span>
                <span>{item.instanceUrl}</span>
              </div>
              <div className="flex py-3 border-b">
                <span className="w-[30%] text-muted-foreground font-semibold shrink-0">
                  Application ID
                </span>
                <span className="truncate">{item.applicationId}</span>
              </div>
              <div className="flex py-3">
                <span className="w-[30%] text-muted-foreground font-semibold shrink-0">
                  Linked repositories
                </span>
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
