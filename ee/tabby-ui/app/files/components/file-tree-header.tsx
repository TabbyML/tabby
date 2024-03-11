'use client'

import { useContext } from 'react'

import useRouterStuff from '@/lib/hooks/use-router-stuff'
import { cn } from '@/lib/utils'
import { IconFolderGit } from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

import { SourceCodeBrowserContext } from './source-code-browser'
import { resolveRepoNameFromPath } from './utils'

interface FileTreeHeaderProps extends React.HTMLAttributes<HTMLDivElement> {}

const FileTreeHeader: React.FC<FileTreeHeaderProps> = ({
  className,
  ...props
}) => {
  const { activePath, fileTreeData, setActivePath, initialized } = useContext(
    SourceCodeBrowserContext
  )
  const { router, searchParams } = useRouterStuff()
  const repoName = resolveRepoNameFromPath(activePath)

  const noIndexedRepo = initialized && !fileTreeData?.length

  const onSelectRepo = (name: string) => {
    setActivePath(name)
  }

  return (
    <div className={cn(className)} {...props}>
      <div className="py-4 font-bold leading-8">Files</div>
      <Select onValueChange={onSelectRepo} value={repoName}>
        <SelectTrigger>
          <SelectValue>
            <div className="flex items-center gap-2">
              <IconFolderGit />
              <span className={repoName ? '' : 'text-muted-foreground'}>
                {repoName || 'Pick a repository'}
              </span>
            </div>
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {noIndexedRepo ? (
            <SelectItem isPlaceHolder value="" disabled>
              No indexed repository
            </SelectItem>
          ) : (
            <>
              {fileTreeData?.map(repo => {
                return (
                  <SelectItem key={repo.fullPath} value={repo.name}>
                    {repo.name}
                  </SelectItem>
                )
              })}
            </>
          )}
        </SelectContent>
      </Select>
    </div>
  )
}

export { FileTreeHeader }
