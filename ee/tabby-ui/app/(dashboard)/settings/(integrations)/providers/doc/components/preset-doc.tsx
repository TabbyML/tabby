'use client'

import React, { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { PresetWebDocumentsQuery } from '@/lib/gql/generates/graphql'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useMutation } from '@/lib/tabby/gql'
import { ArrayElementType } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { IconClose, IconListFilter } from '@/components/ui/icons'
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
    $after: String
    $before: String
    $first: Int
    $last: Int
    $isActive: Boolean!
  ) {
    presetWebDocuments(
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
> & {
  isActive: boolean
}

export default function PresetDocument() {
  const [isActive, setIsActive] = useState('0')
  const [filterPattern, setFilterPattern] = useState<string | undefined>()
  const [debouncedFilterPattern] = useDebounceValue(filterPattern, 200)
  const [list, setList] = useState<ListItem[] | undefined>()
  const [loadingNames, setLoadingNames] = useState<Set<string>>(new Set())
  const inputRef = useRef<HTMLInputElement>(null)

  const [{ data, stale }] = useQuery({
    query: listPresetWebDocuments,
    variables: {
      isActive: isActive === '1'
    }
  })

  const setPresetDocumentActive = useMutation(setPresetDocumentActiveMutation)

  const triggerJobRun = useMutation(triggerJobRunMutation)
  const handleTriggerJobRun = (command: string) => {
    return triggerJobRun({ command }).then(res => {
      if (res?.data?.triggerJobRun) {
        toast.success(
          'The job has been triggered successfully, it may take a few minutes to process.'
        )
        // FIXME getItemByID
      } else {
        toast.error(res?.error?.message || 'Failed to trigger job')
      }
    })
  }

  const onCheckedChange = (name: string, checked: boolean) => {
    if (loadingNames.has(name)) return

    setLoadingNames(prev => {
      const nextSet = new Set(prev)
      nextSet.add(name)
      return nextSet
    })

    // optimistic update
    setList(l =>
      l?.map(o => {
        if (o.node.name === name) {
          return {
            ...o,
            isActive: checked
          }
        }
        return o
      })
    )

    setPresetDocumentActive({
      input: {
        name,
        active: checked
      }
    })
      .then(res => {
        if (!res?.data?.setPresetDocumentActive) {
          const errorMessage = res?.error?.message ?? 'Failed to update'
          toast.error(errorMessage)
          setList(l =>
            l?.map(o => {
              if (o.node.name !== name) {
                return o
              }
              return {
                ...o,
                isActive: !checked
              }
            })
          )
        }
      })
      .finally(() => {
        setLoadingNames(prev => {
          const nextSet = new Set(prev)
          nextSet.delete(name)
          return nextSet
        })
      })
  }

  const clearFilter = () => {
    setFilterPattern('')
    inputRef.current?.focus()
  }

  useEffect(() => {
    setList(
      data?.presetWebDocuments?.edges?.map(o => ({
        ...o,
        isActive: isActive === '1'
      }))
    )
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
          <Select value={isActive} onValueChange={setIsActive}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectGroup>
                <SelectItem value="1">Active</SelectItem>
                <SelectItem value="0">Inactive</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
          <div className="relative">
            <IconListFilter
              className="absolute left-3 top-2.5 cursor-text text-muted-foreground"
              onClick={() => inputRef.current?.focus()}
            />
            <Input
              className="w-50 px-8"
              value={filterPattern}
              onChange={e => setFilterPattern(e.target.value)}
              ref={inputRef}
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
        </div>
      </div>
      <LoadingWrapper loading={!data || stale}>
        <Table className="table-fixed border-b">
          <TableHeader>
            <TableRow>
              <TableHead className="w-[70%]">Name</TableHead>
              <TableHead>Job</TableHead>
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
                        {x.isActive ? (
                          <JobInfoView
                            jobInfo={x.node.jobInfo}
                            onTrigger={async () => {
                              if (x.node?.jobInfo?.command) {
                                handleTriggerJobRun(x.node?.jobInfo.command)
                              }
                            }}
                          />
                        ) : null}
                      </TableCell>
                      <TableCell className="text-right">
                        <Switch
                          checked={x.isActive}
                          onCheckedChange={checked =>
                            onCheckedChange(x.node.name, checked)
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
