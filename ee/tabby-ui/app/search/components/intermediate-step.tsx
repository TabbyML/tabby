import { ReactNode, useEffect, useState } from 'react'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from '@/components/ui/accordion'
import { IconCheckFull, IconSpinner } from '@/components/ui/icons'

export function StepItem({
  isLoading,
  children,
  title,
  defaultOpen,
  isLastItem
}: {
  isLoading: boolean | undefined
  children?: ReactNode
  title: string
  defaultOpen?: boolean
  isLastItem?: boolean
}) {
  const itemName = 'item'
  const [open, setOpen] = useState(!!defaultOpen)
  const hasChildren = !!children

  useEffect(() => {
    if (hasChildren && !open) {
      setOpen(true)
    }
  }, [hasChildren])

  return (
    <div>
      <Accordion
        type="single"
        value={open ? itemName : ''}
        collapsible={hasChildren}
        className="z-10"
        onValueChange={v => {
          setOpen(v === itemName)
        }}
      >
        {/* vertical separator */}
        <AccordionItem value={itemName} className="relative border-0">
          {(!isLastItem || (open && hasChildren)) && (
            <div className="absolute left-2 top-5 block h-full w-0.5 shrink-0 translate-x-px rounded-full bg-muted"></div>
          )}
          <AccordionTrigger
            className="group w-full gap-2 rounded-lg py-1 pl-0.5 pr-2 !no-underline hover:bg-muted/70"
            showChevron={!!children}
          >
            <div className="flex flex-1 items-center gap-4">
              <div className="relative z-10 shrink-0 bg-background group-hover:bg-muted/70">
                {isLoading ? (
                  // <div className="p-0.5">
                  //   <div className="h-3 w-3 rounded-full bg-primary/70" />
                  // </div>
                  <IconSpinner />
                ) : (
                  <IconCheckFull className="h-4 w-4" />
                )}
              </div>
              <span>{title}</span>
            </div>
          </AccordionTrigger>
          {!!children && (
            <AccordionContent className="pb-0 pl-9">
              {children}
            </AccordionContent>
          )}
        </AccordionItem>
      </Accordion>
    </div>
  )
}
