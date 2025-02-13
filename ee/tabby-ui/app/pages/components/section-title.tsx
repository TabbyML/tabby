'use client'

import { HTMLAttributes, useContext } from 'react'

import { cn } from '@/lib/utils'
import { MessageMarkdown } from '@/components/message-markdown'

import { SectionItem } from '../types'
import { PageContext } from './page-context'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  section: SectionItem
}

export function SectionTitle({
  section,
  className,
  ...props
}: QuestionBlockProps) {
  const { pendingSectionIds } = useContext(PageContext)
  const isPending = pendingSectionIds.has(section.id) && !section.content

  return (
    <div className="section-title" id={section.id}>
      <div
        className={cn('flex items-center gap-2 font-semibold', className)}
        id={section.id}
        {...props}
      >
        <MessageMarkdown
          message={section.title}
          contextInfo={undefined}
          supportsOnApplyInEditorV2={false}
          className={cn(
            'text-3xl prose-h2:text-foreground prose-p:mb-1 prose-p:mt-0',
            {
              'text-foreground/50': isPending
            }
          )}
          headline
        />
      </div>
    </div>
  )
}
