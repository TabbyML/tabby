import { Maybe } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { IconCheck, IconFolderGit } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

interface RepoSelectProps {
  // todo rename
  models: Maybe<Array<string>> | undefined
  value: string | undefined
  onChange: (v: string) => void
  isInitializing?: boolean
  // sourceId
  workspaceRepoId?: string
}

export function RepoSelect({
  models,
  value,
  onChange,
  isInitializing
}: RepoSelectProps) {
  const onModelSelect = (v: string) => {
    onChange(v)
  }

  return (
    <LoadingWrapper
      loading={isInitializing}
      fallback={
        <div className="w-full pl-2">
          <Skeleton className="h-3 w-[20%]" />
        </div>
      }
    >
      {!!models?.length && (
        <DropdownMenu>
          <DropdownMenuTrigger>
            <Badge
              variant="outline"
              className="inline-flex h-7 flex-nowrap items-center gap-1 overflow-hidden rounded-md text-sm font-semibold"
            >
              <IconFolderGit />
              {/* FIXME */}
              {/* {value} */}
              TabbyML/tabby
            </Badge>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="bottom"
            align="start"
            className="dropdown-menu max-h-[30vh] min-w-[20rem] overflow-y-auto overflow-x-hidden rounded-md border bg-popover p-2 text-popover-foreground shadow animate-in"
          >
            <DropdownMenuRadioGroup value={value} onValueChange={onChange}>
              {models.map(model => {
                const isSelected = model === value
                return (
                  <DropdownMenuRadioItem
                    onClick={e => {
                      onModelSelect(model)
                      e.stopPropagation()
                    }}
                    value={model}
                    key={model}
                    className="cursor-pointer py-2 pl-3"
                  >
                    <IconCheck
                      className={cn(
                        'mr-2 shrink-0',
                        model === value ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    <span
                      className={cn({
                        'font-medium': isSelected
                      })}
                    >
                      {model}
                    </span>
                  </DropdownMenuRadioItem>
                )
              })}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </LoadingWrapper>
  )
}
