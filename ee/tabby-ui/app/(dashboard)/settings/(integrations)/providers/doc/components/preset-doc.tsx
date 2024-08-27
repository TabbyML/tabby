'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { PresetWebDocumentsQuery } from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { client, useMutation } from '@/lib/tabby/gql'
import { ArrayElementType } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { IconClose, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
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

import { JobInfoView } from '../../components/job-trigger'
import { triggerJobRunMutation } from '../../query'
import { TypeFilter } from './type-filter'

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
  const [isActiveFilter, setIsActiveFilter] = useState('all')
  const [filterPattern, setFilterPattern] = useState<string | undefined>()
  const [debouncedFilterPattern] = useDebounceValue(filterPattern, 200)
  const [list, setList] = useState<ListItem[] | undefined>()
  const [processingIds, setProcessingIds] = useState<Set<string>>(new Set())
  const inputRef = useRef<HTMLInputElement>(null)
  const getIsActiveFromFilter = (filter: string) => {
    if (filter === 'all') return undefined
    return filter === 'active' ? true : false
  }
  const [{ data, stale }] = useQuery({
    query: listPresetWebDocuments,
    variables: {
      isActive: getIsActiveFromFilter(isActiveFilter)
    }
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

  const filteredList = useMemo(() => {
    if (!debouncedFilterPattern) return list
    return (
      list?.filter(item =>
        item.node.name.toLowerCase().includes(debouncedFilterPattern)
      ) ?? []
    )
  }, [debouncedFilterPattern, list])

  return (
    <>
      <div className="my-4 flex justify-between">
        <TypeFilter type="preset" />
        <div className="flex items-center gap-4">
          <div className="relative">
            <IconSearch
              className="absolute left-3 top-2.5 cursor-text text-muted-foreground"
              onClick={() => inputRef.current?.focus()}
            />
            <Input
              className="w-50 px-8"
              value={filterPattern}
              onChange={e => setFilterPattern(e.target.value)}
              ref={inputRef}
              placeholder="Search..."
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
          <Select value={isActiveFilter} onValueChange={setIsActiveFilter}>
            <SelectTrigger className="w-40 gap-2">
              <span>Status:</span>
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectGroup>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </div>
      <LoadingWrapper loading={!data || stale}>
        <Table className="table-fixed border-b">
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead className="w-[100px] lg:w-[200px]">Job</TableHead>
              <TableHead className="w-[100px] text-right">Active</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!filteredList?.length ? (
              <TableRow>
                <TableCell colSpan={3} className="h-[100px] text-center">
                  {!list?.length ? 'No data' : 'No matches data'}
                </TableCell>
              </TableRow>
            ) : (
              <>
                {filteredList?.map(x => {
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
                        />
                      </TableCell>
                    </TableRow>
                  )
                })}
              </>
            )}
          </TableBody>
        </Table>
      </LoadingWrapper>
    </>
  )
}
