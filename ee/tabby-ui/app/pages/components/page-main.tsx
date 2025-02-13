'use client'

import { CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import Link from 'next/link'
import slugify from '@sindresorhus/slugify'
import { AnimatePresence, motion } from 'framer-motion'
import { compact, uniqBy } from 'lodash-es'
import moment from 'moment'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { ERROR_CODE_NOT_FOUND, SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import {
  CreateThreadToPageRunSubscription,
  MoveSectionDirection
} from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { updatePendingThreadId, usePageStore } from '@/lib/stores/page-store'
import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { client, useMutation } from '@/lib/tabby/gql'
import {
  listPages,
  listPageSections,
  listSecuredUsers
} from '@/lib/tabby/query'
import { ExtendedCombinedError } from '@/lib/types'
import { cn } from '@/lib/utils'
import { buttonVariants } from '@/components/ui/button'
import { IconClock, IconFileSearch } from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import LoadingWrapper from '@/components/loading-wrapper'
import { MessageMarkdown } from '@/components/message-markdown'
import NotFoundPage from '@/components/not-found-page'
import { UserAvatar } from '@/components/user-avatar'

import { PageItem, SectionItem } from '../types'
import { Header } from './header'
import { Navbar } from './nav-bar'
import { PageContext } from './page-context'
import { SectionContent } from './section-content'
import { SectionTitle } from './section-title'
import {
  PageSkeleton,
  SectionContentSkeleton,
  SectionsSkeleton,
  SectionTitleSkeleton
} from './skeleton'

const createThreadToPageRunSubscription = graphql(/* GraphQL */ `
  subscription createThreadToPageRun($threadId: ID!) {
    createThreadToPageRun(threadId: $threadId) {
      __typename
      ... on PageCreated {
        id
        authorId
        title
      }
      ... on PageContentDelta {
        delta
      }
      ... on PageContentCompleted {
        id
      }
      ... on PageSectionsCreated {
        sections {
          id
          title
          position
        }
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
      }
      ... on PageCompleted {
        id
      }
    }
  }
`)

const deletePageSectionMutation = graphql(/* GraphQL */ `
  mutation DeletePageSection($sectionId: ID!) {
    deletePageSection(sectionId: $sectionId)
  }
`)

const movePageSectionPositionMutation = graphql(/* GraphQL */ `
  mutation movePageSection($id: ID!, $direction: MoveSectionDirection!) {
    movePageSection(id: $id, direction: $direction)
  }
`)

const PAGE_SIZE = 30

export function Page() {
  const [{ data: meData }] = useMe()
  const { updateUrlComponents, pathname, router } = useRouterStuff()
  const [activePathname, setActivePathname] = useState<string | undefined>()
  const [isPathnameInitialized, setIsPathnameInitialized] = useState(false)
  const pendingThreadId = usePageStore(state => state.pendingThreadId)
  const [mode, setMode] = useState<'edit' | 'view'>(
    pendingThreadId ? 'edit' : 'view'
  )
  // for pending stream sections
  const [pendingSectionIds, setPendingSectionIds] = useState<Set<string>>(
    new Set()
  )

  const [isReady, setIsReady] = useState(!!pendingThreadId)
  // for section skeleton
  const [currentSectionId, setCurrentSectionId] = useState<string | undefined>(
    undefined
  )
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [isShowDemoBanner] = useShowDemoBanner()
  const initializing = useRef(false)
  const { theme } = useCurrentTheme()
  const [pageId, setPageId] = useState<string | undefined>()
  const [page, setPage] = useState<PageItem | undefined>()
  const [sections, setSections] = useState<Array<SectionItem>>()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<ExtendedCombinedError | undefined>()
  const [submitting, setSubmitting] = useState(false)
  const pageIdFromURL = useMemo(() => {
    const regex = /^\/pages\/(.*)/
    if (!activePathname) return undefined

    return activePathname.match(regex)?.[1]?.split('-').pop()
  }, [activePathname])

  const unsubscribeFn = useRef<(() => void) | undefined>()

  const processPageStream = (
    data: CreateThreadToPageRunSubscription['createThreadToPageRun']
  ) => {
    switch (data.__typename) {
      case 'PageCreated':
        const now = new Date().toISOString()
        const nextPage: PageItem = {
          id: data.id,
          authorId: data.authorId,
          title: data.title,
          content: '',
          updatedAt: now,
          createdAt: now
        }
        setPage(nextPage)
        setPageId(data.id)
        // todo remove this
        updatePageURL(nextPage)
        break
      case 'PageContentDelta': {
        setPage(prev => {
          if (!prev) return prev
          return {
            ...prev,
            content: (prev.content || '') + data.delta
          }
        })
        break
      }

      case 'PageContentCompleted': {
        break
      }
      case 'PageSectionsCreated': {
        const nextSections: SectionItem[] = data.sections.map(x => ({
          ...x,
          pageId: pageId as string,
          content: ''
        }))
        setPendingSectionIds(new Set(data.sections.map(x => x.id)))
        setSections(nextSections)
        break
      }
      case 'PageSectionContentDelta': {
        setCurrentSectionId(data.id)
        setSections(prev => {
          const section = prev?.find(x => x.id === data.id)
          if (!section || !prev) {
            return prev
          }
          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                content: (x.content || '') + data.delta
              }
            }
            return x
          })
        })
      }
      case 'PageSectionContentCompleted': {
        setPendingSectionIds(prev => {
          if (!prev.has(data.id)) {
            return prev
          }
          const newSet = new Set(prev)
          newSet.delete(data.id)
          return newSet
        })
        break
      }
      case 'PageCompleted':
        stop.current()
        break
      default:
        break
    }
  }

  const convertThreadToPage = (threadId: string) => {
    const { unsubscribe } = client
      .subscription(createThreadToPageRunSubscription, {
        threadId
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        const value = res.data?.createThreadToPageRun
        if (!value) {
          return
        }

        processPageStream(value)
      })

    return unsubscribe
  }

  const stop = useLatest(() => {
    unsubscribeFn.current?.()
    unsubscribeFn.current = undefined
    setIsLoading(false)
    setPendingSectionIds(new Set())
    setCurrentSectionId(undefined)
    // if (page) {
    //   updatePageURL(page)
    // }
  })

  const deletePageSection = useMutation(deletePageSectionMutation)
  const movePageSectionPosition = useMutation(movePageSectionPositionMutation)

  useEffect(() => {
    if (pageIdFromURL) {
      setPageId(pageIdFromURL)
    }
  }, [pageIdFromURL])

  const [afterCursor, setAfterCursor] = useState<string | undefined>()

  const [{ data: pagesData, error: pageError, fetching: fetchingPage }] =
    useQuery({
      query: listPages,
      variables: {
        ids: [pageIdFromURL] as string[]
      },
      pause: !pageIdFromURL
    })

  useEffect(() => {
    const _page = pagesData?.pages.edges?.[0]?.node
    if (_page) {
      setPage(_page)
    }
  }, [pagesData])

  const [
    {
      data: pageSectionData,
      error: pageSectionsError,
      fetching: fetchingPageSections,
      stale: pageSectionsStale
    }
  ] = useQuery({
    query: listPageSections,
    variables: {
      pageId: pageIdFromURL as string,
      first: PAGE_SIZE,
      after: afterCursor
    },
    pause: !pageIdFromURL || isReady
  })

  useEffect(() => {
    if (pageSectionsStale) return

    if (pageSectionData?.pageSections?.edges?.length) {
      const messages = pageSectionData.pageSections.edges
        .map(x => x.node)
        .slice()
        .sort((a, b) => a.position - b.position)
      setSections(prev => uniqBy([...(prev || []), ...messages], 'id'))
    }

    if (pageSectionData?.pageSections) {
      const hasNextPage = pageSectionData?.pageSections?.pageInfo?.hasNextPage
      const endCursor = pageSectionData?.pageSections.pageInfo.endCursor
      if (hasNextPage && endCursor) {
        setAfterCursor(endCursor)
      } else {
        setIsReady(true)
      }
    }
  }, [pageSectionData])

  const [{ data: authorData, fetching: fetchingAuthor }] = useQuery({
    query: listSecuredUsers,
    variables: {
      ids: compact([page?.authorId])
    },
    pause: !page?.authorId
  })
  const author = authorData?.users?.edges[0]?.node

  const isPageOwner = useMemo(() => {
    if (!meData) return false
    if (!pageIdFromURL) return true

    const page = pagesData?.pages.edges?.[0]
    if (!page) return false

    return meData.me.id === page.node.authorId
  }, [meData, pagesData, pageIdFromURL])

  useEffect(() => {
    if (page?.title) {
      document.title = page.title
    }
  }, [page?.title])

  // FIXME pagesData error handling
  useEffect(() => {
    if (pageSectionsError && !isReady) {
      setIsReady(true)
    }
  }, [pageSectionsError])

  // `/pages` -> `/pages/{slug}-{pageId}`
  const updatePageURL = (page: PageItem) => {
    if (!page) return
    const { title, id } = page
    const firstLine = (title || '').split('\n')[0] ?? ''
    const _title = firstLine.slice(0, SLUG_TITLE_MAX_LENGTH)
    const slug = slugify(_title)
    const slugWithPageId = compact([slug, id]).join('-')

    const path = updateUrlComponents({
      pathname: `/pages/${slugWithPageId}`,
      replace: true
    })

    return location.origin + path
  }

  // for synchronizing the active pathname
  useEffect(() => {
    setActivePathname(pathname)

    if (!isPathnameInitialized) {
      setIsPathnameInitialized(true)
    }
  }, [pathname])

  useEffect(() => {
    const init = () => {
      if (initializing.current) return

      initializing.current = true

      if (pendingThreadId) {
        setIsLoading(true)
        setError(undefined)

        // trigger convert
        unsubscribeFn.current = convertThreadToPage(pendingThreadId)
        updatePendingThreadId(undefined)
        return
      }

      if (!pageId) {
        router.replace('/')
      }
    }

    if (isPathnameInitialized && !pageIdFromURL) {
      init()
    }
  }, [isPathnameInitialized])

  const onDeleteSection = async (sectionId: string) => {
    if (!pageIdFromURL || isLoading) return

    return deletePageSection({ sectionId }).then(data => {
      if (data?.data?.deletePageSection) {
        const nextSections = sections?.filter(x => x.id !== sectionId)
        setSections(nextSections)
      } else {
        toast.error('Failed to delete')
      }
    })
  }

  const reorderSections = (
    sections: SectionItem[],
    sectionId: string,
    direction: MoveSectionDirection
  ) => {
    const currentIndex = sections.findIndex(x => x.id === sectionId)
    if (currentIndex === -1) return sections

    let swapIndex =
      direction === MoveSectionDirection.Up
        ? currentIndex - 1
        : currentIndex + 1

    const updatedSections = sections.map((section, index) => {
      if (index === currentIndex) {
        return {
          ...section,
          position:
            direction === MoveSectionDirection.Up
              ? section.position - 1
              : section.position + 1
        }
      }

      if (index === swapIndex) {
        return {
          ...section,
          position:
            direction === MoveSectionDirection.Up
              ? section.position + 1
              : section.position - 1
        }
      }

      return section
    })

    return updatedSections.slice().sort((a, b) => a.position - b.position)
  }

  const onMoveSectionPosition = async (
    sectionId: string,
    direction: MoveSectionDirection
  ) => {
    if (!pageIdFromURL || isLoading || submitting) return

    setSubmitting(true)
    return movePageSectionPosition({ id: sectionId, direction })
      .then(data => {
        if (data?.data?.movePageSection) {
          setSections(prev => {
            if (!prev) return prev
            // reorder
            return reorderSections(prev, sectionId, direction)
          })
        } else {
          toast.error('Failed to move')
        }
      })
      .finally(() => {
        setSubmitting(false)
      })
  }

  const formatedPageError: ExtendedCombinedError | undefined = useMemo(() => {
    if (!isReady || fetchingPage || !pageIdFromURL) return undefined
    if (pageError || !pagesData?.pages?.edges?.length) {
      return pageError || new Error(ERROR_CODE_NOT_FOUND)
    }
  }, [pagesData, fetchingPage, pageError, isReady, pageIdFromURL])
  const [isFetchingPageSections] = useDebounceValue(
    fetchingPageSections ||
      pageSectionData?.pageSections?.pageInfo?.hasNextPage,
    200
  )

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  if (isReady && (formatedPageError || pageSectionsError)) {
    return (
      <ErrorView
        error={
          (formatedPageError || pageSectionsError) as ExtendedCombinedError
        }
        pageIdFromURL={pageIdFromURL}
      />
    )
  }

  if (!isReady && (isFetchingPageSections || pageSectionsStale)) {
    return (
      <div>
        <Header />
        <PageSkeleton />
      </div>
    )
  }

  return (
    <PageContext.Provider
      value={{
        isLoading,
        isPathnameInitialized,
        isPageOwner,
        mode,
        setMode,
        pendingSectionIds,
        setPendingSectionIds,
        currentSectionId,
        onDeleteSection,
        onMoveSectionPosition
      }}
    >
      <div style={style}>
        <Header pageIdFromURL={pageIdFromURL} streamingDone={!isLoading} />
        <LoadingWrapper loading={!isReady || !page} fallback={<PageSkeleton />}>
          <main className="h-[calc(100%-4rem)] pb-8 lg:pb-0">
            <ScrollArea className="h-full w-full" ref={contentContainerRef}>
              <div className="mx-auto grid grid-cols-4 gap-2 px-4 pb-32 lg:max-w-5xl lg:px-0">
                <div className="relative col-span-3">
                  {/* page title */}
                  <div className="mb-2 mt-8">
                    <LoadingWrapper
                      loading={!page}
                      fallback={<SectionTitleSkeleton />}
                    >
                      <h1 className="text-4xl font-semibold">{page?.title}</h1>
                    </LoadingWrapper>
                    <div className="my-4 flex gap-4 text-sm text-muted-foreground">
                      <LoadingWrapper
                        loading={fetchingAuthor || !page?.authorId}
                      >
                        {!!page && (
                          <>
                            <div className="flex items-center gap-1">
                              <UserAvatar user={author} className="h-6 w-6" />
                              <div>{author?.name}</div>
                            </div>

                            <div className="flex items-center gap-3">
                              <div className="flex items-center gap-0.5">
                                <IconClock />
                                <span>{formatTime(page.createdAt)}</span>
                              </div>
                            </div>
                          </>
                        )}
                      </LoadingWrapper>
                    </div>
                  </div>

                  {/* page content */}
                  <LoadingWrapper
                    // FIXME
                    loading={!page || (isLoading && !page?.content)}
                    fallback={<SectionContentSkeleton />}
                  >
                    <MessageMarkdown
                      message={page?.content ?? ''}
                      supportsOnApplyInEditorV2={false}
                    />
                  </LoadingWrapper>

                  {/* sections */}
                  <LoadingWrapper
                    loading={!page || (isLoading && !sections?.length)}
                    fallback={
                      <div className="my-8 w-full">
                        <SectionsSkeleton />
                      </div>
                    }
                  >
                    <AnimatePresence
                      key={`${isLoading}-${mode}`}
                      initial={false}
                    >
                      {sections?.map((section, index) => {
                        const isSectionGenerating =
                          isLoading && section.id === currentSectionId
                        const enableMoveUp = index !== 0
                        const enableMoveDown = index < sections.length - 1

                        return (
                          <motion.div
                            layout={!isLoading && mode === 'edit'}
                            key={`section_${section.id}`}
                            exit={{ opacity: 0 }}
                          >
                            <SectionTitle
                              className="section-title pt-8"
                              section={section}
                            />
                            <SectionContent
                              section={section}
                              isGenerating={isSectionGenerating}
                              enableMoveUp={enableMoveUp}
                              enableMoveDown={enableMoveDown}
                            />
                          </motion.div>
                        )
                      })}
                    </AnimatePresence>
                  </LoadingWrapper>
                </div>
                <div className="relative col-span-1">
                  <Navbar sections={sections} />
                </div>
              </div>
            </ScrollArea>

            <ButtonScrollToBottom
              className={cn(
                '!fixed !bottom-[5.4rem] !right-4 !top-auto z-40 border-muted-foreground lg:!bottom-[2.85rem]'
              )}
              container={contentContainerRef.current as HTMLDivElement}
              offset={100}
              // On mobile browsers(Chrome & Safari) in dark mode, using `background: hsl(var(--background))`
              // result in `rgba(0, 0, 0, 0)`. To prevent this, explicitly set --background
              style={
                theme === 'dark'
                  ? ({ '--background': '0 0% 12%' } as CSSProperties)
                  : {}
              }
            />
          </main>
        </LoadingWrapper>
      </div>
    </PageContext.Provider>
  )
}

interface ErrorViewProps {
  error: ExtendedCombinedError
  pageIdFromURL?: string
}
function ErrorView({ error, pageIdFromURL }: ErrorViewProps) {
  let title = 'Something went wrong'
  let description = 'Failed to fetch, please refresh the page'

  if (error.message === ERROR_CODE_NOT_FOUND) {
    return <NotFoundPage />
  }

  return (
    <div className="flex h-screen flex-col">
      <Header pageIdFromURL={pageIdFromURL} />
      <div className="flex-1">
        <div className="flex h-full flex-col items-center justify-center gap-2">
          <div className="flex items-center gap-2">
            <IconFileSearch className="h-6 w-6" />
            <div className="text-xl font-semibold">{title}</div>
          </div>
          <div>{description}</div>
          <Link
            href="/"
            onClick={clearHomeScrollPosition}
            className={cn(buttonVariants(), 'mt-4 gap-2')}
          >
            <span>Home</span>
          </Link>
        </div>
      </div>
    </div>
  )
}

function formatTime(time: string) {
  const targetTime = moment(time)

  if (targetTime.isBefore(moment().subtract(1, 'year'))) {
    const timeText = targetTime.format('MMM D, YYYY, h:mm A')
    return timeText
  }

  if (targetTime.isBefore(moment().subtract(1, 'month'))) {
    const timeText = targetTime.format('MMM D, hh:mm A')
    return `${timeText}`
  }

  return `${targetTime.fromNow()}`
}
