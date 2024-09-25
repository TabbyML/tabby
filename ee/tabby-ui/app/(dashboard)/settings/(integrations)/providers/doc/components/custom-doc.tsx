'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { CustomWebDocumentsQuery } from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { client, useMutation } from '@/lib/tabby/gql'
import { userGroupsQuery } from '@/lib/tabby/query'
import { ArrayElementType } from '@/lib/types'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconClose,
  IconListFilter,
  IconPlus,
  IconSearch,
  IconTrash
} from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import LoadingWrapper from '@/components/loading-wrapper'
import { QuickNavPagination } from '@/components/quick-nav-pagination'

import { AccessPolicyView } from '../../components/access-policy-view'
import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const listCustomWebDocuments = graphql(/* GraphQL */ `
  query CustomWebDocuments(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
  ) {
    customWebDocuments(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
    ) {
      edges {
        node {
          url
          name
          id
          sourceId
          jobInfo {
            lastJobRun {
              id
              job
              createdAt
              finishedAt
              exitCode
            }
            command
          }
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
`)

const deleteCustomWebDocumentMutation = graphql(/* GraphQL */ `
  mutation DeleteCustomDocument($id: ID!) {
    deleteCustomDocument(id: $id)
  }
`)

type ListItem = ArrayElementType<
  CustomWebDocumentsQuery['customWebDocuments']['edges']
>

export default function CustomDocument() {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(8)
  const [filterPattern, setFilterPattern] = useState<string | undefined>()
  const [debouncedFilterPattern] = useDebounceValue(filterPattern, 200)
  const [list, setList] = useState<ListItem[] | undefined>()
  const inputRef = useRef<HTMLInputElement>(null)
  const [filterOpen, setFilterOpen] = useState(false)
  const [{ fetching, data, stale }] = useQuery({
    query: listCustomWebDocuments
  })

  const [{ data: userGroupData, fetching: fetchingUserGroups }] = useQuery({
    query: userGroupsQuery
  })

  const clearFilter = () => {
    setFilterPattern('')
    inputRef.current?.focus()
  }

  const deleteCustomWebDocument = useMutation(deleteCustomWebDocumentMutation)

  const handleDeleteCustomDoc = (id: string) => {
    deleteCustomWebDocument({
      id
    })
      .then(res => {
        if (!res?.data?.deleteCustomDocument) {
          const errorMessage = res?.error?.message || 'Failed to delete'
          toast.error(errorMessage)
        } else {
          setList(l => l?.filter(o => o.node.id !== id))
        }
      })
      .catch(e => {
        const errorMessage = e?.message || 'Failed to delete'
        toast.error(errorMessage)
      })
  }

  const getDocumentById = async (id: string) => {
    if (!id) return undefined
    try {
      const res = await client
        .query(listCustomWebDocuments, { ids: [id] })
        .toPromise()
      const record = res?.data?.customWebDocuments?.edges?.[0]
      return record
    } catch (e) {
      return undefined
    }
  }

  const updateDocumentItemById = async (id: string) => {
    try {
      const docItem = await getDocumentById(id)
      if (!docItem?.node?.id || !list?.length) return

      const targetIdx = list.findIndex(o => o.node?.id === docItem.node.id)
      if (targetIdx > -1) {
        setList(prev =>
          prev?.map(o => {
            if (o.node.id === docItem.node.id) {
              return docItem
            } else {
              return o
            }
          })
        )
      }
    } catch (e) {}
  }

  const triggerJobRun = useMutation(triggerJobRunMutation)
  const handleTriggerJobRun = (id: string, command: string) => {
    return triggerJobRun({ command }).then(res => {
      if (res?.data?.triggerJobRun) {
        toast.success(
          'The job has been triggered successfully, it may take a few minutes to process.'
        )

        updateDocumentItemById(id)
      } else {
        toast.error(res?.error?.message || 'Failed to trigger job')
      }
    })
  }

  useEffect(() => {
    setList(data?.customWebDocuments?.edges)
  }, [data])

  const onInputKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ): void => {
    if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
      setFilterOpen(false)
    }
  }

  const filteredList = useMemo(() => {
    if (!debouncedFilterPattern) return list
    return (
      list?.filter(item =>
        item.node.name.toLowerCase().includes(debouncedFilterPattern)
      ) ?? []
    )
  }, [debouncedFilterPattern, list])

  const currentList = useMemo(() => {
    return filteredList?.slice((page - 1) * pageSize, page * pageSize)
  }, [filteredList, page, pageSize])

  // reset pageNo
  useEffect(() => {
    setPage(1)
  }, [debouncedFilterPattern])

  return (
    <>
      <LoadingWrapper loading={fetching}>
        <Table className="min-w-[300px] table-fixed border-b">
          <TableHeader>
            <TableRow>
              <TableHead className="flex items-center gap-1.5">
                Name
                <Popover open={filterOpen} onOpenChange={setFilterOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="relative shrink-0"
                    >
                      <IconListFilter />
                      {!!debouncedFilterPattern && (
                        <div className="absolute right-0 top-1 h-1.5 w-1.5 rounded-full bg-red-400"></div>
                      )}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent align="end" side="right" className="p-1">
                    <div className="relative">
                      <IconSearch
                        className="absolute left-3 top-2.5 cursor-text text-muted-foreground"
                        onClick={() => inputRef.current?.focus()}
                      />
                      <Input
                        size={30}
                        className="w-48 px-8"
                        value={filterPattern}
                        onChange={e => setFilterPattern(e.target.value)}
                        ref={inputRef}
                        placeholder="Search..."
                        onKeyDown={onInputKeyDown}
                      />
                      {filterPattern ? (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute right-3 top-1.5 h-6 w-6 cursor-pointer"
                          onClick={clearFilter}
                        >
                          <IconClose />
                        </Button>
                      ) : null}
                    </div>
                  </PopoverContent>
                </Popover>
                <div>
                  <Link
                    href={`./doc/new`}
                    className={buttonVariants({
                      size: 'icon',
                      variant: 'ghost'
                    })}
                  >
                    <IconPlus />
                  </Link>
                </div>
              </TableHead>
              <TableHead className="w-[140px]">Access</TableHead>
              <TableHead className="w-[180px]">Job</TableHead>
              <TableHead className="w-[60px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!currentList?.length && !fetching ? (
              <TableRow className="hover:bg-background">
                <TableCell colSpan={4} className="h-[100px] text-center">
                  {!list?.length ? (
                    <div className="my-4 flex flex-col items-center gap-4">
                      No data
                      <Link href={`./doc/new`} className={buttonVariants()}>
                        <IconPlus />
                        Add
                      </Link>
                    </div>
                  ) : (
                    'No matches data'
                  )}
                </TableCell>
              </TableRow>
            ) : (
              <>
                {currentList?.map(x => {
                  return (
                    <TableRow key={x.node.id}>
                      <TableCell className="break-all lg:break-words">
                        <p>{x.node.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {x.node.url}
                        </p>
                      </TableCell>
                      <TableCell>
                        <AccessPolicyView
                          sourceId={x.node.sourceId}
                          sourceName={x.node.name}
                          fetchingUserGroups={fetchingUserGroups}
                          userGroups={userGroupData?.userGroups}
                          editable
                        />
                      </TableCell>
                      <TableCell>
                        <JobInfoView
                          jobInfo={x.node.jobInfo}
                          onTrigger={async () => {
                            if (x.node?.jobInfo?.command) {
                              handleTriggerJobRun(
                                x.node.id,
                                x.node?.jobInfo.command
                              )
                            }
                          }}
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="icon"
                          variant="hover-destructive"
                          onClick={() => handleDeleteCustomDoc(x.node.id)}
                        >
                          <IconTrash />
                        </Button>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </>
            )}
          </TableBody>
        </Table>
        <QuickNavPagination
          className="mt-2 flex justify-end"
          page={page}
          pageSize={pageSize}
          showQuickJumper
          totalCount={filteredList?.length ?? 0}
          onChange={(page: number, pageSize: number) => {
            setPage(page)
            setPageSize(pageSize)
          }}
        />
      </LoadingWrapper>
    </>
  )
}
