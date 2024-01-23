'use client'

import React, { PropsWithChildren, useState } from 'react'
import dynamic from 'next/dynamic'
import { has } from 'lodash-es'
import useSWRImmutable from 'swr/immutable'

import { useHealth } from '@/lib/hooks/use-health'
import useRouterStuff from '@/lib/hooks/use-router-stuff'
import fetcher from '@/lib/tabby/fetcher'
import { cn } from '@/lib/utils'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup
} from '@/components/ui/resizable'

import { RepositoriesFileTree, TFileTreeNode } from './file-tree'

const SourceCodeEditor = dynamic(() => import('./source-code-editor'), {
  ssr: false
})

type TCodeMap = Record<string, string>
type TFileMetaMap = Record<string, TFileMeta>
type TCodeTag = {
  range: TRange
  name_range: TRange
  line_range: TRange
  is_definition: boolean
  syntax_type_name: string
  utf16_column_range: TRange
  span: TPointRange
}
type TPoint = { row: number; column: number }
type TPointRange = { start: TPoint; end: TPoint }
type TRange = { start: number; end: number }
type TFileMeta = {
  git_url: string
  filepath: string
  language: string
  max_line_length: number
  avg_line_length: number
  alphanum_fraction: number
  tags: TCodeTag[]
}

type SourceCodeBrowserContextValue = {
  codeMap: Record<string, string>
  setCodeMap: React.Dispatch<React.SetStateAction<TCodeMap>>
  fileMetaMap: TFileMetaMap
  setFileMetaMap: React.Dispatch<React.SetStateAction<TFileMetaMap>>
  activePath: string | undefined
  setActivePath: React.Dispatch<React.SetStateAction<string | undefined>>
}

type SourceCodeBrowserProviderProps = {}

const SourceCodeBrowserContext =
  React.createContext<SourceCodeBrowserContextValue>(
    {} as SourceCodeBrowserContextValue
  )

const SourceCodeBrowserContextProvider: React.FC<
  PropsWithChildren<SourceCodeBrowserProviderProps>
> = ({ children }) => {
  const [activePath, setActivePath] = React.useState<string>()
  const [codeMap, setCodeMap] = useState<TCodeMap>({})
  const [fileMetaMap, setFileMetaMap] = useState<TFileMetaMap>({})

  return (
    <SourceCodeBrowserContext.Provider
      value={{
        codeMap,
        setCodeMap,
        fileMetaMap,
        setFileMetaMap,
        activePath,
        setActivePath
      }}
    >
      {children}
    </SourceCodeBrowserContext.Provider>
  )
}

interface SourceCodeBrowserProps {
  className?: string
}

const SourceCodeBrowserRenderer: React.FC<SourceCodeBrowserProps> = ({
  className
}) => {
  const hea = useHealth()
  const { searchParams, updateSearchParams } = useRouterStuff()
  const defaultRepositoryName = searchParams.get('repo')?.toString()
  const defaultBasename = searchParams.get('path')?.toString()
  const [repositoryName, setRepositoryName] = React.useState<string>(
    defaultRepositoryName ?? ''
  )
  const [fileResolver, setFileResolver] = React.useState<string>(
    defaultBasename ?? ''
  )
  const { activePath, setActivePath, codeMap, setCodeMap, setFileMetaMap } =
    React.useContext(SourceCodeBrowserContext)
  const { data: fileContent } = useSWRImmutable(
    fileResolver && repositoryName
      ? `/repositories/${repositoryName}/resolve/${fileResolver}`
      : null,
    (url: string) => fetcher(url, { format: 'text' })
  )

  const { data: fileMeta } = useSWRImmutable(
    fileContent && fileResolver && repositoryName
      ? `/repositories/${repositoryName}/meta/${fileResolver}`
      : null,
    fetcher
  )

  const onSelectTreeNode = (
    treeNode: TFileTreeNode,
    repositoryName: string
  ) => {
    const path = `${repositoryName}/${treeNode.file.basename}`
    const isFile = treeNode.file.kind === 'file'
    if (isFile) {
      setRepositoryName(repositoryName)
      setActivePath(path)
      if (!has(codeMap, path)) {
        setFileResolver(treeNode.file.basename)
      }

      updateSearchParams({
        set: {
          repo: repositoryName,
          path: treeNode.file.basename
        },
        replace: true
      })
    }
  }

  React.useEffect(() => {
    if (defaultBasename && defaultRepositoryName) {
      setActivePath(`${defaultRepositoryName}/${defaultBasename}`)
    }
  }, [])

  React.useEffect(() => {
    if (fileContent && activePath) {
      setCodeMap(map => ({
        ...map,
        [activePath]: fileContent
      }))
    }
  }, [fileContent])

  React.useEffect(() => {
    if (fileMeta && activePath) {
      setFileMetaMap(map => ({
        ...map,
        [activePath]: fileMeta
      }))
    }
  }, [fileMeta])

  return (
    <ResizablePanelGroup direction="horizontal" className={cn(className)}>
      <ResizablePanel defaultSize={20} minSize={20}>
        <div className="h-full overflow-hidden py-2 pl-4">
          <RepositoriesFileTree
            className="h-full overflow-y-auto overflow-x-hidden pr-4"
            onSelectTreeNode={onSelectTreeNode}
            activePath={activePath}
            defaultBasename={defaultBasename}
            defaultRepository={defaultRepositoryName}
          />
        </div>
      </ResizablePanel>
      <ResizableHandle className="w-1 hover:bg-primary active:bg-primary" />
      <ResizablePanel defaultSize={80} minSize={30}>
        <SourceCodeEditor
          className={`flex h-full ${activePath ? 'block' : 'hidden'}`}
        />
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}

const SourceCodeBrowser: React.FC<SourceCodeBrowserProps> = props => {
  return (
    <SourceCodeBrowserContextProvider>
      <SourceCodeBrowserRenderer className="source-code-browser" {...props} />
    </SourceCodeBrowserContextProvider>
  )
}

export {
  SourceCodeBrowserContext,
  SourceCodeBrowser,
  type TCodeTag,
  type TRange,
  type TFileMeta
}
