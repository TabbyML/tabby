'use client'

import React from 'react'

import { useScrollTop } from '@/lib/hooks/use-scroll-top'
import { Input } from '@/components/ui/input'

import { FileTree, TFileTreeNode } from './file-tree'
import { FileTreeHeader } from './file-tree-header'
import { SourceCodeBrowserContext } from './source-code-browser'
import { generateEntryPath } from './utils'

interface FileTreePanelProps extends React.HTMLAttributes<HTMLDivElement> {
  fetchingTreeEntries: boolean
}

export const FileTreePanel: React.FC<FileTreePanelProps> = ({
  fetchingTreeEntries
}) => {
  const {
    activePath,
    updateActivePath,
    expandedKeys,
    updateFileMap,
    toggleExpandedKey,
    initialized,
    fileTreeData,
    fileMap,
    activeRepo,
    activeRepoRef
  } = React.useContext(SourceCodeBrowserContext)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const scrollTop = useScrollTop(containerRef, 200)
  const onSelectTreeNode = (treeNode: TFileTreeNode) => {
    const nextPath = generateEntryPath(
      activeRepo,
      activeRepoRef?.name as string,
      treeNode.file.basename,
      treeNode.file.kind
    )
    updateActivePath(nextPath)
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* <FileTreeHeader className="shrink-0 px-4 pb-3" /> */}
      <div className="px-4 mt-3.5">
        <Input placeholder="Go to file" />
      </div>
      {scrollTop > 0 && (
        <div className="h-0 border-b shadow-[0px_3px_8px_rgba(0,0,0,0.3)] dark:shadow-[0px_3px_8px_rgba(255,255,255,0.3)]"></div>
      )}
      <div className="flex-1 py-2 overflow-y-auto px-4" ref={containerRef}>
        <FileTree
          onSelectTreeNode={onSelectTreeNode}
          activePath={activePath}
          fileMap={fileMap}
          updateFileMap={updateFileMap}
          expandedKeys={expandedKeys}
          toggleExpandedKey={toggleExpandedKey}
          initialized={initialized}
          fileTreeData={fileTreeData}
          fetchingTreeEntries={fetchingTreeEntries}
        />
      </div>
    </div>
  )
}
