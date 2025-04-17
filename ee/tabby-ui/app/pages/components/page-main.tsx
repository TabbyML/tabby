'use client'

import { CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import slugify from '@sindresorhus/slugify'
import { AnimatePresence, motion } from 'framer-motion'
import { compact, flatten, omit, uniqBy } from 'lodash-es'
import { ImperativePanelHandle } from 'react-resizable-panels'
import { toast } from 'sonner'
import { useQuery } from 'urql'

import { ERROR_CODE_NOT_FOUND, SLUG_TITLE_MAX_LENGTH } from '@/lib/constants'
import {
  useEnableDeveloperMode,
  useEnableSearchPages
} from '@/lib/experiment-flags'
import {
  CreatePageRunSubscription,
  CreatePageSectionRunSubscription,
  MoveSectionDirection
} from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useDebounceValue } from '@/lib/hooks/use-debounce'
import { useLatest } from '@/lib/hooks/use-latest'
import { useMe } from '@/lib/hooks/use-me'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { clearPendingThread, usePageStore } from '@/lib/stores/page-store'
import { client, useMutation } from '@/lib/tabby/gql'
import {
  contextInfoQuery,
  listPages,
  listPageSections,
  listSecuredUsers
} from '@/lib/tabby/query'
import { ExtendedCombinedError } from '@/lib/types'
import { cn, isCodeSourceContext, nanoid } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconBug, IconStop } from '@/components/ui/icons'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Skeleton } from '@/components/ui/skeleton'
import { ButtonScrollToBottom } from '@/components/button-scroll-to-bottom'
import { BANNER_HEIGHT, useShowDemoBanner } from '@/components/demo-banner'
import { DevPanel } from '@/components/dev-panel'
import LoadingWrapper from '@/components/loading-wrapper'
import { SourceIcon } from '@/components/source-icon'
import { UserAvatar } from '@/components/user-avatar'

import {
  createPageRunSubscription,
  createPageSectionRunSubscription,
  createThreadToPageRunSubscription,
  deletePageSectionMutation,
  movePageSectionPositionMutation
} from '../lib/query'
import { formatTime } from '../lib/utils'
import {
  DebugData,
  PageItem,
  SectionDebugDataItem,
  SectionItem
} from '../types'
import { ErrorView } from './error-view'
import { Header } from './header'
import { Navbar } from './nav-bar'
import { NewPageForm } from './new-page-form'
import { NewSectionForm } from './new-section-form'
import { PageContent } from './page-content'
import { PageContext } from './page-context'
import { PageTitle } from './page-title'
import { SectionContent } from './section-content'
import { SectionTitle } from './section-title'
import {
  PageSkeleton,
  SectionContentSkeleton,
  SectionsSkeleton,
  SectionTitleSkeleton
} from './skeleton'

const PAGE_SIZE = 30

type PageRunItem = CreatePageRunSubscription['createPageRun']

export function Page() {
  const [{ data: meData }] = useMe()
  const [{ data: contextInfoData, fetching: fetchingContextInfo }] = useQuery({
    query: contextInfoQuery
  })
  const [enableDeveloperMode] = useEnableDeveloperMode()
  const [enableSearchPages] = useEnableSearchPages()
  const { updateUrlComponents, pathname, router } = useRouterStuff()
  const [activePathname, setActivePathname] = useState<string | undefined>()
  const [isPathnameInitialized, setIsPathnameInitialized] = useState(false)
  const pendingThreadId = usePageStore(state => state.pendingThreadId)
  const pendingThreadTitle = usePageStore(state => state.pendingThreadTitle)
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
  const [pageCompleted, setPageCompleted] = useState(true)
  const contentContainerRef = useRef<HTMLDivElement>(null)
  const [isShowDemoBanner] = useShowDemoBanner()
  const initializing = useRef(false)
  const { theme } = useCurrentTheme()
  const [pageId, setPageId] = useState<string | undefined>()
  const [page, setPage] = useState<PageItem | undefined>()
  const [debugData, setDebugData] = useState<DebugData | undefined>()
  const [sections, setSections] = useState<Array<SectionItem>>()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<ExtendedCombinedError | undefined>()
  const [submitting, setSubmitting] = useState(false)
  const [isGeneratingPageTitle, setIsGeneratingPageTitle] = useState(false)
  const [devPanelOpen, setDevPanelOpen] = useState(false)
  const devPanelRef = useRef<ImperativePanelHandle>(null)
  const [devPanelSize, setDevPanelSize] = useState(45)
  const prevDevPanelSize = useRef(devPanelSize)
  const [stopGenerateVisible, setStopGenerateVisible] = useState(false)

  const pageIdFromURL = useMemo(() => {
    const regex = /^\/pages\/(.*)/
    if (!activePathname) return undefined
    const id = activePathname.match(regex)?.[1]?.split('-').pop()
    return id === 'new' ? undefined : id
  }, [activePathname])

  const isNew = useMemo(() => {
    if (!isPathnameInitialized) return false

    return activePathname === '/pages/new'
  }, [isPathnameInitialized])

  const unsubscribeFn = useRef<(() => void) | undefined>()
  const stop = useLatest(() => {
    unsubscribeFn.current?.()
    unsubscribeFn.current = undefined
    setIsLoading(false)
    setPendingSectionIds(new Set())
    setCurrentSectionId(undefined)
    setPageCompleted(true)
    setStopGenerateVisible(false)
  })

  const onStopGenerating = () => {
    stop.current()

    if (isLoading) {
      // remove empty sections
      setSections(prev => {
        if (!prev) return prev
        return prev.filter(x => !!x.content)
      })
      setPendingSectionIds(new Set())
    }
  }

  const updateDebugData = (data: DebugData | undefined | null) => {
    if (!data) return

    setDebugData(prev => {
      if (!prev) return omit(data, '__typename')

      if ('generateSectionTitlesMessages' in data) {
        return {
          ...prev,
          generateSectionTitlesMessages: compact(
            flatten([
              prev.generateSectionTitlesMessages,
              data.generateSectionTitlesMessages
            ])
          )
        }
      } else {
        return {
          ...prev,
          ...omit(data, '__typename')
        }
      }
    })
  }
  const updateSectionDebugData = (
    debugData: SectionDebugDataItem | null | undefined,
    sectionId: string | undefined
  ) => {
    if (!debugData || !sectionId) return
    setDebugData(prev => {
      if (!prev) {
        return {
          sections: [
            {
              id: sectionId,
              ...debugData
            }
          ]
        }
      }

      const sections = prev.sections || []
      let target = sections.find(s => s.id === sectionId)
      if (target) {
        return {
          ...prev,
          sections: sections.map(x => {
            if (x.id === sectionId) {
              return {
                ...x,
                ...debugData
              }
            }
            return x
          })
        }
      } else {
        return {
          ...prev,
          sections: sections.concat([{ id: sectionId, ...debugData }])
        }
      }
    })
  }

  // `/pages` -> `/pages/{slug}-{pageId}`
  const updatePageURL = (page: { id: string; title: string }) => {
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

  const processPageRunItemStream = (data: PageRunItem) => {
    switch (data.__typename) {
      case 'PageCreated': {
        setPageId(data.id)
        setPage(prev => {
          if (!prev) return prev

          return {
            ...prev,
            id: data.id,
            title: data.title,
            authorId: data.authorId
          }
        })
        updateDebugData(data.debugData)
        setIsGeneratingPageTitle(false)
        setStopGenerateVisible(true)
        updatePageURL(data)
        break
      }
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
        updateDebugData(data.debugData)
        break
      }
      case 'PageSectionsCreated': {
        const nextSections: SectionItem[] = data.sections.map(x => ({
          ...x,
          pageId: pageId as string,
          content: '',
          attachments: {
            code: [],
            codeFileList: null,
            doc: []
          }
        }))
        setPendingSectionIds(new Set(data.sections.map(x => x.id)))
        setSections(nextSections)
        updateDebugData(data.debugData)
        break
      }
      case 'PageSectionAttachmentCodeFileList': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  ...x.attachments,
                  codeFileList: data.codeFileList
                }
              }
            } else {
              return x
            }
          })
        })
        break
      }
      case 'PageSectionAttachmentCode': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  ...x.attachments,
                  code: data.codes.map(x => ({
                    ...x.code,
                    extra: { scores: x.scores }
                  }))
                }
              }
            }

            return x
          })
        })
        // group by id
        const debugData: SectionDebugDataItem | undefined = data.debugData
          ? { attachmentCodeQuery: data.debugData }
          : undefined
        updateSectionDebugData(debugData, data.id)
        break
      }
      case 'PageSectionAttachmentDoc': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  ...x.attachments,
                  doc: data.doc.map(x => ({
                    ...x.doc,
                    extra: { score: x.score }
                  }))
                }
              }
            }
            return x
          })
        })

        // group by id
        const debugData: SectionDebugDataItem | undefined = data.debugData
          ? { attachmentDocQuery: data.debugData }
          : undefined
        updateSectionDebugData(debugData, data.id)
        break
      }
      case 'PageSectionContentDelta': {
        setCurrentSectionId(data.id)
        setSections(prev => {
          if (!prev) {
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
        break
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

        // group by id
        const debugData = data.debugData?.generateSectionContentMessages
          ? {
              generateSectionContentMessages:
                data.debugData.generateSectionContentMessages
            }
          : undefined
        updateSectionDebugData(debugData, data.id)
        break
      }
      case 'PageCompleted':
        stop.current()
        break
      default:
        break
    }
  }

  const processSectionRunStream = (
    data: CreatePageSectionRunSubscription['createPageSectionRun']
  ) => {
    switch (data.__typename) {
      case 'PageSectionCreated': {
        const { id, title, position } = data
        setCurrentSectionId(id)
        setPendingSectionIds(new Set([id]))
        setSections(prev => {
          if (!prev) return prev
          const _sections = prev.slice(0, -1)
          return [
            ..._sections,
            {
              id,
              title,
              position,
              content: '',
              pageId: pageId as string,
              attachments: {
                code: [],
                codeFileList: null,
                doc: []
              }
            }
          ]
        })
        updateDebugData(data.debugData)
        setStopGenerateVisible(true)
        break
      }
      case 'PageSectionContentDelta': {
        const { delta, id } = data
        setSections(prev => {
          if (!prev) return prev
          return prev.map(x => {
            if (x.id === id) {
              return {
                ...x,
                content: x.content + delta
              }
            }
            return x
          })
        })
        break
      }
      case 'PageSectionAttachmentCodeFileList': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  ...x.attachments,
                  codeFileList: data.codeFileList
                }
              }
            } else {
              return x
            }
          })
        })
        break
      }
      case 'PageSectionAttachmentCode': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  code: data.codes.map(x => ({
                    ...x.code,
                    extra: { scores: x.scores }
                  })),
                  codeFileList: x.attachments.codeFileList
                }
              }
            } else {
              return x
            }
          })
        })
        // group by id
        const debugData: SectionDebugDataItem | undefined = data.debugData
          ? { attachmentCodeQuery: data.debugData }
          : undefined
        updateSectionDebugData(debugData, data.id)
        break
      }
      case 'PageSectionAttachmentDoc': {
        setSections(prev => {
          if (!prev || !prev.length) return prev

          return prev.map(x => {
            if (x.id === data.id) {
              return {
                ...x,
                attachments: {
                  ...x.attachments,
                  doc: data.doc.map(x => ({
                    ...x.doc,
                    extra: { score: x.score }
                  }))
                }
              }
            }
            return x
          })
        })
        // group by id
        const debugData: SectionDebugDataItem | undefined = data.debugData
          ? { attachmentDocQuery: data.debugData }
          : undefined
        updateSectionDebugData(debugData, data.id)
        break
      }
      case 'PageSectionContentCompleted': {
        // group by id
        const debugData: SectionDebugDataItem | undefined = data.debugData
          ?.generateSectionContentMessages?.length
          ? {
              generateSectionContentMessages:
                data.debugData.generateSectionContentMessages
            }
          : undefined
        updateSectionDebugData(debugData, data.id)

        stop.current()
        break
      }
    }
  }

  const appendNewSection = async (title: string) => {
    if (!pageId) return

    const tempSectionId = nanoid()
    setIsLoading(true)
    setError(undefined)
    setSections(prev => {
      const lastPosition = prev?.[prev.length - 1]?.position || 0
      const newSection: SectionItem = {
        id: tempSectionId,
        title,
        pageId,
        content: '',
        position: lastPosition + 1,
        attachments: {
          code: [],
          codeFileList: null,
          doc: []
        }
      }

      if (!prev) return [newSection]
      return [...prev, newSection]
    })
    setPendingSectionIds(new Set([tempSectionId]))
    setCurrentSectionId(tempSectionId)

    const { unsubscribe } = client
      .subscription(createPageSectionRunSubscription, {
        input: {
          pageId,
          titlePrompt: title,
          docQuery: {
            sourceIds: compact([
              page?.codeSourceId,
              enableSearchPages.value ? 'page' : undefined
            ]),
            content: title,
            searchPublic: false
          },
          debugOption: enableDeveloperMode?.value
            ? {
                returnChatCompletionRequest: true,
                returnQueryRequest: true
              }
            : undefined
        }
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }

        const value = res?.data?.createPageSectionRun
        if (!value) {
          return
        }
        processSectionRunStream(value)
      })

    unsubscribeFn.current = unsubscribe
  }

  const convertThreadToPage = (threadId: string, threadTitle: string) => {
    const now = new Date().toISOString()
    const tempId = nanoid()
    const nextPage: PageItem = {
      id: tempId,
      authorId: '',
      title: threadTitle,
      content: '',
      updatedAt: now,
      createdAt: now
    }
    setPage(nextPage)
    setIsLoading(true)
    setPageId(tempId)
    setIsGeneratingPageTitle(true)
    setError(undefined)
    setPageCompleted(false)

    const { unsubscribe } = client
      .subscription(createThreadToPageRunSubscription, {
        input: {
          threadId,
          debugOption: enableDeveloperMode?.value
            ? {
                returnChatCompletionRequest: true,
                returnQueryRequest: true
              }
            : undefined
        }
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

        processPageRunItemStream(value)
      })

    return unsubscribe
  }

  const createPage = async ({
    titlePrompt,
    codeSourceId
  }: {
    titlePrompt: string
    codeSourceId?: string
  }) => {
    const now = new Date().toISOString()
    const tempId = nanoid()
    const nextPage: PageItem = {
      id: tempId,
      authorId: '',
      title: titlePrompt,
      content: '',
      codeSourceId,
      updatedAt: now,
      createdAt: now
    }
    setPage(nextPage)
    setPageId(tempId)
    setIsLoading(true)
    setIsGeneratingPageTitle(true)
    setPageCompleted(false)

    const { unsubscribe } = client
      .subscription(createPageRunSubscription, {
        input: {
          titlePrompt,
          codeQuery: codeSourceId
            ? {
                sourceId: codeSourceId,
                content: titlePrompt
              }
            : null,
          docQuery: {
            sourceIds: compact([
              codeSourceId,
              enableSearchPages.value ? 'page' : undefined
            ]),
            content: titlePrompt,
            searchPublic: false
          },
          debugOption: enableDeveloperMode?.value
            ? {
                returnChatCompletionRequest: true,
                returnQueryRequest: true
              }
            : undefined
        }
      })
      .subscribe(res => {
        if (res?.error) {
          setIsLoading(false)
          setError(res.error)
          unsubscribe()
          return
        }
        const value = res.data?.createPageRun
        if (!value) {
          return
        }

        processPageRunItemStream(value)
      })

    unsubscribeFn.current = unsubscribe
  }

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
      pause: !pageIdFromURL || isNew
    })

  useEffect(() => {
    const _page = pagesData?.pages.edges?.[0]?.node
    if (_page && !isNew) {
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
    pause: !pageIdFromURL || isReady || isNew
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
    if (!pageIdFromURL || !page?.authorId) return true

    return meData.me.id === page?.authorId
  }, [meData, page?.authorId, pageIdFromURL])

  const repository = useMemo(() => {
    if (!page?.codeSourceId) return undefined

    const target = contextInfoData?.contextInfo?.sources.find(
      x => isCodeSourceContext(x.sourceKind) && x.sourceId === page.codeSourceId
    )
    return target
  }, [page?.codeSourceId, contextInfoData])

  useEffect(() => {
    if (devPanelOpen) {
      devPanelRef.current?.expand()
      devPanelRef.current?.resize(devPanelSize)
    } else {
      devPanelRef.current?.collapse()
    }
  }, [devPanelOpen])

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

  // for synchronizing the active pathname
  useEffect(() => {
    setActivePathname(pathname)

    if (!isPathnameInitialized) {
      setIsPathnameInitialized(true)
    }
  }, [pathname])

  // for refresh /pages/new
  useEffect(() => {
    if (isPathnameInitialized && !!page && activePathname === '/pages/new') {
      window.location.reload()
    }
  }, [activePathname])

  useEffect(() => {
    const init = () => {
      if (initializing.current) return

      initializing.current = true

      if (pendingThreadId) {
        // trigger convert
        unsubscribeFn.current = convertThreadToPage(
          pendingThreadId,
          pendingThreadTitle ?? ''
        )
        clearPendingThread()
        return
      }

      if (isNew) {
        setMode('edit')
      }

      if (!pageId && !isNew) {
        router.replace('/')
      }
    }

    if (isPathnameInitialized && !pageIdFromURL) {
      init()
    }
  }, [isPathnameInitialized])

  useEffect(() => {
    return () => {
      unsubscribeFn.current?.()
    }
  }, [])

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

  const onUpdateSections = (id: string, values: Partial<SectionItem>) => {
    if (!id) return

    setSections(prev => {
      if (!prev) return prev

      return prev.map(x => {
        if (x.id === id) {
          return {
            ...x,
            ...values
          }
        }
        return x
      })
    })
  }

  const onPanelLayout = (sizes: number[]) => {
    if (sizes?.[1]) {
      setDevPanelSize(sizes[1])
    }
  }

  const onToggleFullScreen = (fullScreen: boolean) => {
    let nextSize = prevDevPanelSize.current
    if (fullScreen) {
      nextSize = 100
    } else if (nextSize === 100) {
      nextSize = 45
    }
    devPanelRef.current?.resize(nextSize)
    setDevPanelSize(nextSize)
    prevDevPanelSize.current = devPanelSize
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

  if (isReady && (formatedPageError || pageSectionsError || error)) {
    return (
      <ErrorView
        error={
          (formatedPageError ||
            pageSectionsError ||
            error) as ExtendedCombinedError
        }
        pageIdFromURL={pageIdFromURL}
      />
    )
  }

  if (!isNew && !isReady && (isFetchingPageSections || pageSectionsStale)) {
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
        contextInfo: contextInfoData?.contextInfo,
        fetchingContextInfo,
        pendingSectionIds,
        setPendingSectionIds,
        currentSectionId,
        pageIdFromURL,
        isNew,
        onDeleteSection,
        onMoveSectionPosition,
        enableDeveloperMode: enableDeveloperMode.value,
        devPanelOpen,
        setDevPanelOpen
      }}
    >
      <div style={style}>
        <ResizablePanelGroup direction="vertical" onLayout={onPanelLayout}>
          <ResizablePanel>
            <Header pageIdFromURL={pageIdFromURL} streamingDone={!isLoading} />
            <LoadingWrapper
              loading={!isNew && (!isReady || !page)}
              fallback={<PageSkeleton />}
              delay={0}
            >
              <main className="h-[calc(100%-4rem)] pb-8 lg:pb-0">
                <ScrollArea className="h-full w-full" ref={contentContainerRef}>
                  <div className="mx-auto grid grid-cols-4 gap-2 px-4 pb-32 lg:max-w-5xl lg:px-0">
                    {isNew && !page ? (
                      <div className="col-span-4 mt-8 rounded-xl border pl-1 pr-3 pt-2 ring-2 ring-transparent transition-colors focus-within:ring-ring focus-visible:ring-ring">
                        <NewPageForm onSubmit={createPage} />
                      </div>
                    ) : (
                      <>
                        <div className="relative col-span-3">
                          {/* page title */}
                          <div className="mb-2 mt-8">
                            <div
                              className={cn('flex items-center gap-2', {
                                'mb-4': !!repository || enableDeveloperMode
                              })}
                            >
                              {!!repository && (
                                <div className="inline-flex items-center gap-1 rounded-lg bg-accent px-2 py-1 text-xs font-medium text-accent-foreground">
                                  <SourceIcon
                                    kind={repository.sourceKind}
                                    className="h-3.5 w-3.5 shrink-0"
                                  />
                                  <span className="truncate">
                                    {repository.sourceName}
                                  </span>
                                </div>
                              )}
                              {enableDeveloperMode && (
                                <Button
                                  size="icon"
                                  variant="ghost"
                                  onClick={() => {
                                    setDevPanelOpen(true)
                                  }}
                                >
                                  <IconBug />
                                </Button>
                              )}
                            </div>
                            <LoadingWrapper
                              loading={!page}
                              fallback={<SectionTitleSkeleton />}
                            >
                              <PageTitle
                                page={page}
                                isGeneratingPageTitle={isGeneratingPageTitle}
                                onUpdate={title => {
                                  setPage(p => {
                                    if (!p) return p
                                    return { ...p, title }
                                  })
                                }}
                              />
                            </LoadingWrapper>
                            <div className="my-4 flex gap-4 text-sm">
                              <LoadingWrapper
                                loading={fetchingAuthor || !page?.authorId}
                                fallback={<Skeleton />}
                              >
                                <div className="flex items-center gap-2">
                                  <UserAvatar
                                    user={author}
                                    className="h-8 w-8"
                                  />
                                  <div className="pt-0.5">
                                    <div className="text-sm leading-none">
                                      {author?.name}
                                    </div>
                                    <span className="text-xs leading-none text-muted-foreground">
                                      {formatTime(page?.createdAt)}
                                    </span>
                                  </div>
                                </div>
                              </LoadingWrapper>
                            </div>
                          </div>

                          {/* page content */}
                          <LoadingWrapper
                            loading={!page || (isLoading && !page?.content)}
                            fallback={<SectionContentSkeleton />}
                          >
                            <PageContent
                              page={page}
                              onUpdate={content => {
                                setPage(p => {
                                  if (!p) return p
                                  return { ...p, content }
                                })
                              }}
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
                                const enableMoveDown =
                                  index < sections.length - 1
                                return (
                                  <motion.div
                                    layout={
                                      !isLoading && mode === 'edit'
                                        ? 'position'
                                        : false
                                    }
                                    key={`section_${section.id}`}
                                    exit={{ opacity: 0 }}
                                    className="space-y-2"
                                    transition={
                                      isLoading
                                        ? { duration: 0 }
                                        : { duration: 0.3 }
                                    }
                                  >
                                    <SectionTitle
                                      className="pt-12 prose-p:leading-tight"
                                      section={section}
                                      onUpdate={title => {
                                        onUpdateSections(section.id, { title })
                                      }}
                                    />
                                    <SectionContent
                                      section={section}
                                      isGenerating={isSectionGenerating}
                                      enableMoveUp={enableMoveUp}
                                      enableMoveDown={enableMoveDown}
                                      onUpdate={content => {
                                        onUpdateSections(section.id, {
                                          content
                                        })
                                      }}
                                      enableDeveloperMode={
                                        enableDeveloperMode.value
                                      }
                                    />
                                  </motion.div>
                                )
                              })}
                            </AnimatePresence>
                            {/* append section */}
                            {isPageOwner &&
                              mode === 'edit' &&
                              pageCompleted && (
                                <NewSectionForm
                                  onSubmit={appendNewSection}
                                  disabled={!pageId || isLoading}
                                  className="mt-10"
                                />
                              )}
                          </LoadingWrapper>
                        </div>
                        <div className="relative col-span-1">
                          <Navbar sections={sections} />
                        </div>
                      </>
                    )}
                  </div>
                </ScrollArea>
                {stopGenerateVisible && (
                  <div className="fixed bottom-16 w-full">
                    <div className="mx-auto grid grid-cols-4 gap-2 lg:max-w-5xl">
                      <motion.div
                        className="col-span-3 flex h-px justify-center overflow-y-visible"
                        animate={{ opacity: 100, y: 0 }}
                        initial={{ opacity: 0, y: 20 }}
                        transition={{
                          ease: 'easeInOut',
                          duration: 0.4,
                          delay: 0.5
                        }}
                      >
                        <Button
                          onClick={onStopGenerating}
                          variant="outline"
                          className="gap-2 bg-background"
                        >
                          <IconStop />
                          stop generating
                        </Button>
                      </motion.div>
                    </div>
                  </div>
                )}
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
          </ResizablePanel>
          <ResizableHandle
            className={cn(
              'hidden !h-[4px] border-none bg-background shadow-[0px_-4px_4px_rgba(0,0,0,0.2)] hover:bg-blue-500 active:bg-blue-500 dark:shadow-[0px_-4px_4px_rgba(255,255,255,0.2)]',
              devPanelOpen && 'block'
            )}
          />
          <ResizablePanel
            collapsible
            collapsedSize={0}
            defaultSize={0}
            ref={devPanelRef}
            onCollapse={() => setDevPanelOpen(false)}
            className="z-50"
          >
            <DevPanel
              onClose={() => setDevPanelOpen(false)}
              value={debugData}
              isFullScreen={devPanelSize === 100}
              onToggleFullScreen={onToggleFullScreen}
              scrollOnUpdate={false}
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </PageContext.Provider>
  )
}
