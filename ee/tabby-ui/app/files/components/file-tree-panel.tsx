'use client'

import React from 'react'

import { useScrollTop } from '@/lib/hooks/use-scroll-top'

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
    activeEntryInfo
  } = React.useContext(SourceCodeBrowserContext)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const scrollTop = useScrollTop(containerRef, 200)
  const onSelectTreeNode = (treeNode: TFileTreeNode) => {
    const nextPath = generateEntryPath(
      activeRepo,
      activeEntryInfo.rev,
      treeNode.file.basename,
      treeNode.file.kind
    )
    updateActivePath(nextPath)
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <FileTreeHeader className="shrink-0 px-4 pb-3" />
      {scrollTop > 0 && (
        <div className="h-0 border-b shadow-[0px_3px_8px_rgba(0,0,0,0.3)] dark:shadow-[0px_3px_8px_rgba(255,255,255,0.3)]"></div>
      )}
      <div className="flex-1 overflow-y-auto px-4" ref={containerRef}>
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
