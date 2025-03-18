'use client'

import { HTMLAttributes, useContext, useEffect, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { MessageMarkdown } from '@/components/message-markdown'

import { SectionItem } from '../types'
import { MessageContentForm } from './message-content-form'
import { PageContext } from './page-context'

const updatePageSectionTitleMutation = graphql(/* GraphQL */ `
  mutation updatePageSectionTitle($input: UpdatePageSectionTitleInput!) {
    updatePageSectionTitle(input: $input)
  }
`)

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  section: SectionItem
  onUpdate: (title: string) => void
}

export function SectionTitle({
  section,
  className,
  onUpdate,
  ...props
}: QuestionBlockProps) {
  const { pendingSectionIds, mode, isLoading } = useContext(PageContext)
  const [showForm, setShowForm] = useState(false)
  const isEditing = mode === 'edit'
  const isPending = pendingSectionIds.has(section.id) && !section.content
  const updatePageSectionTitle = useMutation(updatePageSectionTitleMutation)

  const handleSubmitTitleChange = async (title: string) => {
    const result = await updatePageSectionTitle({
      input: {
        id: section.id,
        title
      }
    })

    if (result?.data?.updatePageSectionTitle) {
      onUpdate(title)
      setShowForm(false)
    } else {
      let error = result?.error
      return error
    }
  }

  useEffect(() => {
    if (!isEditing && showForm) {
      setShowForm(true)
    }
  }, [isEditing])

  return (
    <div className={cn('section-title', className)} id={section.id}>
      {showForm ? (
        <MessageContentForm
          message={section.title}
          onSubmit={handleSubmitTitleChange}
          onCancel={() => setShowForm(false)}
          inputClassName="text-3xl font-semibold"
        />
      ) : (
        <div
          className={cn('flex items-baseline gap-2 font-semibold')}
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
          {isEditing && !isLoading && (
            <Button
              size="icon"
              className="shrink-0 rounded-full shadow-none"
              onClick={e => setShowForm(true)}
              variant="secondary"
            >
              <IconEdit />
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
