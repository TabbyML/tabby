'use client'

import { Maybe } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'

import LoadingWrapper from '../loading-wrapper'
import { Button } from '../ui/button'
import { IconBox, IconCheck } from '../ui/icons'
import { Skeleton } from '../ui/skeleton'

interface ModelSelectProps {
  models: Maybe<Array<string>> | undefined
  value: string | undefined
  onChange: (v: string) => void
  isInitializing?: boolean
  triggerClassName?: string
}

export function ModelSelect({
  models,
  value,
  onChange,
  isInitializing,
  triggerClassName
}: ModelSelectProps) {
  const onSelectModel = (v: string) => {
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
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              className={cn(
                'gap-2 px-1.5 py-1 text-foreground/90',
                triggerClassName
              )}
            >
              <IconBox />
              {value}
            </Button>
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
                      onSelectModel(model)
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
