'use client'

import {
  CSSProperties,
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import Link from 'next/link'
import { some, uniqBy } from 'lodash-es'
import { nanoid } from 'nanoid'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { ERROR_CODE_NOT_FOUND } from '@/lib/constants'
import { graphql } from '@/lib/gql/generates'
import {
  ConvertThreadToPageSubscription,
  ListPageSectionsQuery,
  ListPagesQuery,
  Message
} from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { updatePendingThreadId, usePageStore } from '@/lib/stores/page-store'
import { clearHomeScrollPosition } from '@/lib/stores/scroll-store'
import { client, useMutation } from '@/lib/tabby/gql'
import { listPages, listPageSections } from '@/lib/tabby/query'
import { ExtendedCombinedError } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Button, buttonVariants } from '@/components/ui/button'
import {
  IconClock,
  IconFileSearch,
  IconList,
  IconPlus,
  IconSheet,
  IconStop
} from '@/components/ui/icons'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import LoadingWrapper from '@/components/loading-wrapper'
import { MessageMarkdown } from '@/components/message-markdown'
import NotFoundPage from '@/components/not-found-page'
import { MyAvatar } from '@/components/user-avatar'

import { Header } from './header'
import { MessagesSkeleton } from './messages-skeleton'
import { Navbar } from './nav-bar'
import { PageContext } from './page-context'
import { SectionContent } from './section-content'
import SectionForm from './section-form'
import { SectionTitle } from './section-title'

const convertThreadToPageSubscription = graphql(/* GraphQL */ `
  subscription ConvertThreadToPage($threadId: ID!) {
    convertThreadToPage(threadId: $threadId) {
      __typename
      ... on PageCreated {
        id
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
        }
      }
      ... on PageSectionContentDelta {
        id
        delta
      }
      ... on PageSectionContentCompleted {
        id
      }
    }
  }
`)

export type SectionItem = Omit<
  Message,
  '__typename' | 'updatedAt' | 'createdAt'
> & {
  pageId?: string
  sectionId: string
  error?: string
}

const deletePageSectionMutation = graphql(/* GraphQL */ `
  mutation DeletePageSection($sectionId: ID!) {
    deletePageSection(sectionId: $sectionId)
  }
`)

const addPageSectionMutation = graphql(/* GraphQL */ `
  mutation AddPageSection($input: AddPageSectionInput!) {
    addPageSection(input: $input)
  }
`)

const PAGE_SIZE = 30

const TEMP_MSG_ID_PREFIX = '_temp_msg_'
const tempNanoId = () => `${TEMP_MSG_ID_PREFIX}${nanoid()}`

export function Page() {
  const [{ data: meData }] = useMe()
  const { pathname } = useRouterStuff()
  const [activePathname, setActivePathname] = useState<string | undefined>()
  const [isPathnameInitialized, setIsPathnameInitialized] = useState(false)
  const [mode, setMode] = useState<'edit' | 'view'>('view')
  // for pending stream sections
  const [pendingSectionIds, setPendingSectionIds] = useState<Set<string>>(
    new Set()
  )
  const [stopButtonVisible, setStopButtonVisible] = useState(true)

  const pendingThreadId = usePageStore(state => state.pendingThreadId)

  const [isReady, setIsReady] = useState(!!pendingThreadId)
  // for section skeleton
  const [currentSectionId, setCurrentSectionId] = useState<string | undefined>(
    undefined
  )
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [showSectionInput, setShowSectionInput] = useState(false)
  const [isShowDemoBanner] = useShowDemoBanner()
  const initializing = useRef(false)
  const { theme } = useCurrentTheme()
  const [pageId, setPageId] = useState<string | undefined>()
  const [page, setPage] = useState<
    ListPagesQuery['pages']['edges'][0]['node'] | undefined
  >()
  const [sections, setSections] =
    useState<Array<ListPageSectionsQuery['pageSections']['edges'][0]['node']>>()
  const [isLoading, setIsLoading] = useState(false)
  const pageIdFromURL = useMemo(() => {
    const regex = /^\/page\/(.*)/
    if (!activePathname) return undefined

    return activePathname.match(regex)?.[1]?.split('-').pop()
  }, [activePathname])

  const unsubscribeFn = useRef<(() => void) | undefined>()

  const processPageStream = (
    data: ConvertThreadToPageSubscription['convertThreadToPage']
  ) => {
    switch (data.__typename) {
      case 'PageCreated':
        // FIXME should return other information of a page?
        setPageId(data.id)
        if (!page) {
          const now = new Date().toISOString()
          setPage({
            id: data.id,
            authorId: meData?.me.id || '',
            updatedAt: now,
            createdAt: now
          })
        }
        break
      case 'PageContentDelta':
        if (page) {
          setPage(prev => ({
            ...prev,
            content: (prev?.content || '') + data.delta
          }))
        }
        break
      case 'PageContentCompleted': {
        // todo
        break
      }
      case 'PageSectionsCreated': {
        const nextSections = data.sections.map(x => ({
          ...x,
          pageId: pageId as string,
          content: ''
        }))
        setPendingSectionIds(new Set(data.sections.map(x => x.id)))
        setSections(nextSections)
        break
      }
      case 'PageSectionContentDelta': {
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
        const newSet = new Set(pendingSectionIds)
        if (newSet.has(data.id)) {
          newSet.delete(data.id)
          setPendingSectionIds(newSet)
        }
        break
      }
      // todo pagecompleted
    }
  }

  const convertThreadToPage = (threadId: string) => {
    const { unsubscribe } = client
      .subscription(convertThreadToPageSubscription, {
        threadId
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          // todo error
          // setError(res.error)
          unsubscribe()
          return
        }

        const value = res.data?.convertThreadToPage
        if (!value) {
          return
        }

        // if (value?.__typename === 'ThreadAssistantMessageCompleted') {
        //   stop.current()
        // }

        if (value?.__typename === 'PageCreated') {
          if (value.id !== pageId) {
            setPageId(value.id)
          }
        }

        processPageStream(value)
      })

    return unsubscribe
  }

  const onPageCompleted = () => {}

  const stop = useLatest(() => {
    unsubscribeFn.current?.()
    unsubscribeFn.current = undefined
    setIsLoading(false)

    onPageCompleted?.()
  })

  const deletePageSection = useMutation(deletePageSectionMutation)
  const addPageSection = useMutation(addPageSectionMutation)

  const onUpdateSectionContent = async (
    content: string
  ): Promise<ExtendedCombinedError | undefined> => {
    // todo
    return
  }

  useEffect(() => {
    if (pageIdFromURL) {
      setPageId(pageIdFromURL)
    }
  }, [pageIdFromURL])

  const [afterCursor, setAfterCursor] = useState<string | undefined>()

  const [{ data: pagesData }] = useQuery({
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
      const messages = pageSectionData.pageSections.edges.slice()
      setSections(prev => uniqBy([...(prev || []), ...messages], 'cursor'))
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

  const isPageOwner = useMemo(() => {
    if (!meData) return false
    if (!pageIdFromURL) return true

    const page = pagesData?.pages.edges?.[0]
    if (!page) return false

    return meData.me.id === page.node.authorId
  }, [meData, pagesData, pageIdFromURL])

  // todo title slug and update title
  useEffect(() => {
    if (page?.title) {
      document.title = page.title
    }
  }, [page?.title])

  // todo pagesData error
  useEffect(() => {
    if (pageSectionsError && !isReady) {
      setIsReady(true)
    }
  }, [pageSectionsError])

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
        convertThreadToPage(pendingThreadId)
        updatePendingThreadId(undefined)
      }
    }

    if (isPathnameInitialized && !pageIdFromURL) {
      init()
    }
  }, [isPathnameInitialized])

  // Display the input field with a delayed animatio
  useEffect(() => {
    if (isReady) {
      setTimeout(() => {
        setShowSectionInput(true)
      }, 300)
    }
  }, [isReady])

  // Delay showing the stop button
  const showStopTimeoutId = useRef<number>()

  const isLoadingRef = useLatest(isLoading)

  useEffect(() => {
    if (isLoadingRef.current) {
      showStopTimeoutId.current = window.setTimeout(() => {
        if (!isLoadingRef.current) return
        setStopButtonVisible(true)

        // Scroll to the bottom
        const container = contentContainerRef?.current
        if (container) {
          container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
          })
        }
      }, 300)
    }

    if (!isLoadingRef.current) {
      setStopButtonVisible(false)
    }

    return () => {
      window.clearTimeout(showStopTimeoutId.current)
    }
  }, [isLoading])

  const onAddSection = (title: string) => {
    if (!pageIdFromURL) return
    addPageSection({
      input: {
        title,
        pageId: pageIdFromURL
      }
    })
  }

  const onDeleteSection = (sectionId: string) => {
    if (!pageIdFromURL) return

    deletePageSection({ sectionId }).then(data => {
      if (data?.data?.deletePageSection) {
        const nextSections = sections?.filter(x => x.node.id !== sectionId)
        setSections(nextSections)
      } else {
        toast.error('Failed to delete')
      }
    })
  }

  const formatedThreadError = undefined
  const [isFetchingPageSections] = useDebounceValue(
    fetchingPageSections ||
      pageSectionData?.pageSections?.pageInfo?.hasNextPage,
    200
  )

  const style = isShowDemoBanner
    ? { height: `calc(100vh - ${BANNER_HEIGHT})` }
    : { height: '100vh' }

  if (isReady && (formatedThreadError || pageSectionsError)) {
    return (
      <ErrorView
        error={
          (formatedThreadError || pageSectionsError) as ExtendedCombinedError
        }
        pageIdFromURL={pageIdFromURL}
      />
    )
  }

  if (!isReady && (isFetchingPageSections || pageSectionsStale)) {
    return (
      <div>
        <Header />
        <div className="mx-auto mt-24 w-full space-y-10 px-4 pb-32 lg:max-w-5xl lg:px-0">
          <MessagesSkeleton />
          <MessagesSkeleton />
        </div>
      </div>
    )
  }

  return (
    <PageContext.Provider
      value={{
        isLoading,
        onAddSection,
        isPathnameInitialized,
        onDeleteSection,
        isPageOwner,
        onUpdateSectionContent,
        mode,
        setMode
      }}
    >
      <div style={style}>
        <Header pageIdFromURL={pageIdFromURL} streamingDone={!isLoading} />
        <LoadingWrapper
          loading={!isReady}
          fallback={
            <div className="mx-auto mt-24 w-full space-y-10 px-4 pb-32 lg:max-w-4xl lg:px-0">
              <MessagesSkeleton />
              <MessagesSkeleton />
            </div>
          }
        >
          <main className="h-[calc(100%-4rem)] pb-8 lg:pb-0">
            <ScrollArea className="h-full w-full" ref={contentContainerRef}>
              <div className="mx-auto grid grid-cols-4 gap-2 px-4 pb-32 lg:max-w-5xl lg:px-0">
                <div className="col-span-3 relative">
                  {/* page title */}
                  <div className="mb-2 mt-4">
                    <h1 className="text-4xl font-semibold">{page?.title}</h1>
                    <div className="my-4 flex gap-4 text-sm text-muted-foreground">
                      {/* FIXME fetch author */}
                      <div className="flex items-center gap-1">
                        <MyAvatar className="h-6 w-6" />
                        <div>{meData?.me?.name}</div>
                      </div>

                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-0.5">
                          <IconClock />
                          <span>{page?.createdAt}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* page content */}
                  <MessageMarkdown
                    message={page?.content ?? ''}
                    supportsOnApplyInEditorV2={false}
                  />

                  {/* sections */}
                  <div className="flex flex-col">
                    {sections?.map(({ node: section }, index) => {
                      const isLastSection = index === sections.length - 1
                      const isSectionLoading =
                        isLoading && section.id === currentSectionId
                      return (
                        <Fragment key={section.id}>
                          <SectionTitle
                            className="section-title pt-8"
                            message={section}
                          />
                          <SectionContent
                            className="pb-8"
                            message={section}
                            isLoading={isSectionLoading}
                          />
                        </Fragment>
                      )
                    })}
                  </div>
                  {mode === 'edit' && (
                    <div className="rounded-xl border p-2">
                      <div className="flex items-center gap-1">
                        <Button size="icon" variant="ghost">
                          <IconList />
                        </Button>
                        <Button size="icon" variant="ghost">
                          <IconSheet />
                        </Button>
                      </div>
                      <SectionForm
                        onSearch={onAddSection}
                        className="min-h-[5rem] border-0 lg:max-w-5xl"
                        placeholder="What is the section about?"
                        isLoading={isLoading}
                      />
                    </div>
                  )}
                  <div
                    className={cn(
                      'pointer-events-none fixed bottom-5 z-30 w-full',
                      {
                        'opacity-100 translate-y-0': showSectionInput,
                        'opacity-0 translate-y-10': !showSectionInput
                      }
                    )}
                    style={Object.assign(
                      { transition: 'all 0.35s ease-out' },
                      theme === 'dark'
                        ? ({ '--background': '0 0% 12%' } as CSSProperties)
                        : {}
                    )}
                  >
                    <div className="pointer-events-auto">
                      <div
                        className={cn('absolute flex items-center gap-4')}
                        style={isPageOwner ? { top: '-2.5rem' } : undefined}
                      >
                        {stopButtonVisible && (
                          <Button
                            className="bg-background"
                            variant="outline"
                            onClick={() => stop.current()}
                          >
                            <IconStop className="mr-2" />
                            Stop generating
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
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
  let description =
    'Failed to fetch, please refresh the page or create a new page'

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
            <IconPlus />
            <span>New Page</span>
          </Link>
        </div>
      </div>
    </div>
  )
}

function formatThreadRunErrorMessage(error?: ExtendedCombinedError) {
  if (!error) return 'Failed to fetch'

  if (error.message === '401') {
    return 'Unauthorized'
  }

  if (
    some(error.graphQLErrors, o => o.extensions?.code === ERROR_CODE_NOT_FOUND)
  ) {
    return `The thread has expired or does not exist.`
  }

  return error.message || 'Failed to fetch'
}
