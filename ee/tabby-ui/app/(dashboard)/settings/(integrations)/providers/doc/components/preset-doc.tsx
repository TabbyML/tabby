'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { go as fuzzy } from 'fuzzysort'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { PresetWebDocumentsQuery } from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { client, useMutation } from '@/lib/tabby/gql'
import { ArrayElementType } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { IconClose, IconListFilter, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import { Switch } from '@/components/ui/switch'
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

import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'

const listPresetWebDocuments = graphql(/* GraphQL */ `
  query PresetWebDocuments(
    $ids: [ID!]
    $after: String
    $before: String
    $first: Int
    $last: Int
    $isActive: Boolean
  ) {
    presetWebDocuments(
      ids: $ids
      after: $after
      before: $before
      first: $first
      last: $last
      isActive: $isActive
    ) {
      edges {
        node {
          id
          name
          isActive
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

const setPresetDocumentActiveMutation = graphql(/* GraphQL */ `
  mutation SetPresetDocumentActive($input: SetPresetDocumentActiveInput!) {
    setPresetDocumentActive(input: $input)
  }
`)

type ListItem = ArrayElementType<
  PresetWebDocumentsQuery['presetWebDocuments']['edges']
>

export default function PresetDocument() {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(8)
  const [filterPattern, setFilterPattern] = useState<string | undefined>()
  const [debouncedFilterPattern] = useDebounceValue(filterPattern, 200)
  const [list, setList] = useState<ListItem[] | undefined>()
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set())
  const inputRef = useRef<HTMLInputElement>(null)
  const [filterOpen, setFilterOpen] = useState(false)
  const [{ data, stale }] = useQuery({
    query: listPresetWebDocuments
  })

  const setPresetDocumentActive = useMutation(setPresetDocumentActiveMutation)

  const getDocumentById = async (id: string) => {
    if (!id) return undefined
    try {
      const res = await client
        .query(listPresetWebDocuments, { ids: [id] })
        .toPromise()
      const record = res?.data?.presetWebDocuments?.edges?.[0]
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

  const onCheckedChange = (id: string, checked: boolean) => {
    if (processingIds.has(id)) return

    setProcessingIds(prev => {
      const nextSet = new Set(prev)
      nextSet.add(id)
      return nextSet
    })

    // optimistic update
    setList(l =>
      l?.map(o => {
        if (o.node.id === id) {
          return {
            ...o,
            node: {
              ...o.node,
              isActive: checked
            }
          }
        }
        return o
      })
    )

    setPresetDocumentActive({
      input: {
        id,
        active: checked
      }
    })
      .then(res => {
        if (!res?.data?.setPresetDocumentActive) {
          const errorMessage = res?.error?.message ?? 'Failed to update'
          toast.error(errorMessage)
          setList(l =>
            l?.map(o => {
              if (o.node.id !== id) {
                return o
              }
              return {
                ...o,
                node: {
                  ...o.node,
                  isActive: !checked
                }
              }
            })
          )
        }
      })
      .finally(() => {
        setProcessingIds(prev => {
          const nextSet = new Set(prev)
          nextSet.delete(id)
          return nextSet
        })
        updateDocumentItemById(id)
      })
  }

  const clearFilter = () => {
    setFilterPattern('')
    inputRef.current?.focus()
  }

  useEffect(() => {
    setList(data?.presetWebDocuments?.edges)
  }, [data])

  const onInputKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ): void => {
    if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
      setFilterOpen(false)
    }
  }

  const filteredList = useMemo(() => {
    if (!debouncedFilterPattern || !list?.length) return list ?? []

    const result = fuzzy(debouncedFilterPattern, list, {
      key: item => item.node.name
    })
    return result.map(o => o.obj)
  }, [debouncedFilterPattern, list])

  const currentList = useMemo(() => {
    return filteredList?.slice((page - 1) * pageSize, page * pageSize)
  }, [filteredList, page, pageSize])

  // reset pageNo
  useEffect(() => {
    setPage(1)
  }, [debouncedFilterPattern])

  return (
    <div className="min-h-[30.5rem]">
      <LoadingWrapper loading={!data || stale}>
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
              </TableHead>
              <TableHead className="w-[100px] lg:w-[200px]">Job</TableHead>
              <TableHead className="w-[100px] text-right"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!currentList?.length ? (
              <TableRow>
                <TableCell colSpan={3} className="h-[100px] text-center">
                  {!list?.length ? 'No data' : 'No matches data'}
                </TableCell>
              </TableRow>
            ) : (
              <>
                {currentList?.map(x => {
                  return (
                    <TableRow key={x.node.id}>
                      <TableCell className="break-all lg:break-words">
                        {x.node.name}
                      </TableCell>
                      <TableCell>
                        {x.node.isActive ? (
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
                        ) : null}
                      </TableCell>
                      <TableCell className="text-right">
                        <Switch
                          checked={x.node.isActive}
                          onCheckedChange={checked =>
                            onCheckedChange(x.node.id, checked)
                          }
                          className="my-1"
                        />
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
    </div>
  )
}
