'use client'

import React from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { find, uniqueId } from 'lodash-es'
import { toast } from 'sonner'
import useLocalStorage from 'use-local-storage'

import { Button } from '@/components/ui/button'
import { CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/dialog'
import { IconGitHub, IconTrash } from '@/components/ui/icons'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'
import { ListSkeleton } from '@/components/skeleton'

import LinkRepositoryForm from './new-repository-form'
import { UpdateProviderForm } from './provider-detail-form'

const DetailPage: React.FC = () => {
  const searchParams = useSearchParams()
  const router = useRouter()
  const id = searchParams.get('id')
  const [mockProviders, setMockProviders] = useLocalStorage<Array<any> | null>(
    'mock-gitops-data',
    null
  )
  const [mockRepo, setMockRepo] = useLocalStorage<Array<any> | null>(
    'mock-linked-repo',
    null
  )
  const data = find(mockProviders, item => String(item.id) === id)
  const [open, setOpen] = React.useState(false)

  const onCreated = (values: any) => {
    const currentRepos = mockRepo || []
    setMockRepo([...currentRepos, { ...values, id: uniqueId() }])
    toast.success('Created successfully')
    setOpen(false)
  }

  const onDeleteRepo = (id: string) => {
    setMockRepo(mockRepo?.filter(item => item.id !== id) ?? [])
  }

  const onDeleteProvider = () => {
    setMockProviders(mockProviders?.filter(item => item.id !== id))
    router.replace('/settings/gitops')
  }

  return (
    <div>
      <CardHeader className="pl-0 pt-0">
        <CardTitle className="flex items-center justify-between">
          <span>Provider information</span>
          <div className="flex items-center gap-2 text-base">
            <IconGitHub className="h-6 w-6" />
            GitHub.com
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="pl-0">
        <LoadingWrapper fallback={<ListSkeleton />}>
          <UpdateProviderForm
            defaultValues={data}
            onDelete={onDeleteProvider}
            onBack={() => {
              router.push('/settings/gitops')
            }}
          />
        </LoadingWrapper>
      </CardContent>

      <CardHeader className="mt-8 pl-0 pt-0">
        <CardTitle className="flex items-center justify-between">
          <span>Linked repositories</span>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent>
              <DialogHeader className="gap-3">
                <DialogTitle>New Repository</DialogTitle>
                <DialogDescription>
                  Add a new repository link to this provider.
                </DialogDescription>
              </DialogHeader>
              <LinkRepositoryForm
                onCancel={() => setOpen(false)}
                onCreated={onCreated}
              />
            </DialogContent>
            <DialogTrigger asChild>
              <Button>New repository</Button>
            </DialogTrigger>
          </Dialog>
        </CardTitle>
      </CardHeader>
      <CardContent className="pl-0">
        <LoadingWrapper>
          <LinkedRepoTable data={mockRepo ?? []} onDelete={onDeleteRepo} />
        </LoadingWrapper>
      </CardContent>
    </div>
  )
}

const LinkedRepoTable: React.FC<{
  data: Array<any>
  onDelete: (id: string) => void
}> = ({ data, onDelete }) => {
  return (
    <Table className="mt-4">
      <TableHeader>
        <TableRow>
          <TableHead className="w-[25%]">Repository name</TableHead>
          <TableHead className="w-[45%]">Git url</TableHead>
          <TableHead></TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {data?.length ? (
          <>
            {data?.map(x => {
              return (
                <TableRow key={x.id}>
                  <TableCell>{x.name}</TableCell>
                  <TableCell>{x.gitUrl}</TableCell>
                  <TableCell className="flex justify-end">
                    <div className="flex gap-1">
                      <Button
                        size="icon"
                        variant="hover-destructive"
                        onClick={e => onDelete(x.id)}
                      >
                        <IconTrash />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              )
            })}
          </>
        ) : (
          <TableRow>
            <TableCell colSpan={3} className="h-[100px] text-center">
              No linked repositories yet.
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  )
}

export default DetailPage
