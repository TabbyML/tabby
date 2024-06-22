'use client'

import React, { FormEventHandler, useContext, useEffect, useState } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { GrepTextOrBase64, RepositoryKind } from '@/lib/gql/generates/graphql'
import { client } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { Form } from '@/components/ui/form'
import { IconClose, IconSearch } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepositoryInfoFromPath } from './utils'

interface GrepSubMatch {
  byteStart: number
  byteEnd: number
}

interface GrepLine {
  line: GrepTextOrBase64
  byteOffset: number
  lineNumber: number
  subMatches: GrepSubMatch[]
}

interface GrepFile {
  path: string
  lines: GrepLine[]
}

type FormValues = z.infer<typeof formSchema>

interface GlobalSearchProps {
  searchTabIsActive: boolean
  activateSearchTab: () => void
  deactivateSearchTab: () => void
}

const formSchema = z.object({
  // PLACEHOLDER
  // TODO: Adjust for context
  to: z.string().email('Invalid email address')
})

const globalSearchQuery = graphql(/* GraphQL */ `
  query GlobalSearch($id: ID!, $kind: RepositoryKind!, $query: String!) {
    repositoryGrep(kind: $kind, id: $id, query: $query) {
      path
      lines {
        line {
          text
          base64
        }
        byteOffset
        lineNumber
        subMatches {
          bytesStart
          bytesEnd
        }
      }
    }
  }
`)

const GLOBAL_SEARCH_SHORTCUT = 's'

const GlobalSearch: React.FC<GlobalSearchProps> = ({
  ...props
}: GlobalSearchProps) => {
  const { activePath, activeRepo } = useContext(SourceCodeBrowserContext)

  const { repositoryKind } = resolveRepositoryInfoFromPath(activePath)

  const repoId = activeRepo?.id

  /**
   *
   */
  const inputRef = React.useRef<HTMLInputElement>(null)

  /**
   * The current search value. Set `onInput` or by the
   * setup effect when the URL has a query parameter.
   */
  const [value, setValue] = useState('')

  /**
   * Check if the URL has a query parameter and conditionally
   * set the value of the search input.
   */
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search)
    const query = urlParams.get('q')

    if (query) {
      setValue(query)
    }
  }, [])

  /**
   * The async task to fetch the search results from the server.
   * Runs with every input change. Sets the value of the results
   */
  const onInput: FormEventHandler<HTMLInputElement> = e => {
    const query = e.currentTarget.value
    setValue(query)
  }

  /**
   *
   */
  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as Element
      const tagName = target?.tagName?.toLowerCase()
      if (
        tagName === 'input' ||
        tagName === 'textarea' ||
        tagName === 'select'
      ) {
        return
      }

      if (event.key === GLOBAL_SEARCH_SHORTCUT) {
        event.preventDefault()
        inputRef.current?.focus()
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  /**
   * The async task to fetch the search results from the server.
   * Called by the `onInput` event handler when the input value changes.
   */
  const search = async (query: string) => {
    const { data } = (await client
      .query(globalSearchQuery, {
        id: repoId as string,
        kind: repositoryKind as RepositoryKind,
        query,
        pause: !repoId || !repositoryKind
      })
      // FIXME: Wrong types
      .toPromise()) as unknown as { data: { repositoryGrep: GrepFile[] } }
    if (!data) return
  }

  /**
   *
   */
  const clearInput = () => {
    setValue('')
    inputRef.current?.focus()
  }

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema)
  })

  // Placeholder
  const submitForm: FormEventHandler<HTMLFormElement> = e => {
    e.preventDefault()
    search(value)
  }

  return (
    // TODO: Componentize the search input
    <Form {...form}>
      <form
        onSubmit={submitForm}
        className={`w-full flex items-center h-14 ${
          props.searchTabIsActive ? '' : ''
        }`}
      >
        <div className="relative w-full">
          <Input
            type="search"
            placeholder="Search..."
            className="w-full"
            value={value}
            ref={inputRef}
            onInput={onInput}
            onFocus={props.activateSearchTab}
            className="text-center pr-32"
          />
          <div className="absolute right-2 top-0 flex h-full items-center">
            {value ? (
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 cursor-pointer"
                onClick={() => {
                  clearInput()
                }}
              >
                <IconClose />
              </Button>
            ) : (
              <kbd
                className="rounded-md border bg-secondary/50 px-1.5 text-xs leading-4 text-muted-foreground shadow-[inset_-0.5px_-1.5px_0_hsl(var(--muted))]"
                onClick={() => {
                  inputRef.current?.focus()
                }}
              >
                {GLOBAL_SEARCH_SHORTCUT}
              </kbd>
            )}
            <div className="border-l-border border-l flex items-center ml-2 pl-2">
              <Button
                variant="ghost"
                className="h-6 w-6 "
                size="icon"
                type="submit"
              >
                <IconSearch />
              </Button>
            </div>
          </div>
        </div>
      </form>
    </Form>
  )
}

export { GlobalSearch }
