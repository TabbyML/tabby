'use client'

import { HTMLAttributes, useContext, useEffect, useState } from 'react'

import { cn } from '@/lib/utils'
import { MessageMarkdown } from '@/components/message-markdown'

import { SectionItem } from '../types'
import { PageContext } from './page-context'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { MessageContentForm } from './message-content-form'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  section: SectionItem
}

export function SectionTitle({
  section,
  className,
  ...props
}: QuestionBlockProps) {
  const { pendingSectionIds, mode } = useContext(PageContext)
  const [showForm, setShowForm] = useState(false)
  const isEditing = mode === 'edit'
  const isPending = pendingSectionIds.has(section.id) && !section.content

  const handleSubmit = async (message: string) => {
    // todo submit content page
    setShowForm(false)
  }

  useEffect(() => {
    if (!isEditing && showForm) {
      setShowForm(true)
    }
  }, [isEditing])

  return (
    <div className={cn("section-title", className)} id={section.id}>
      {showForm ? (
        <MessageContentForm
          message={section.title}
          onSubmit={handleSubmit}
          onCancel={() => setShowForm(false)}
        />
      ) : (
        <div
          className={cn('font-semibold')}
          id={section.id}
          {...props}
        >
          <MessageMarkdown
            message={section.title}
            contextInfo={undefined}
            supportsOnApplyInEditorV2={false}
            className={cn(
              'line-block text-3xl prose-h2:text-foreground prose-p:mb-1 prose-p:mt-0',
              {
                'text-foreground/50': isPending
              }
            )}
            headline
          />
          {isEditing && (
            <Button className='inline-block' onClick={e => setShowForm(true)}>
              <IconEdit />
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
