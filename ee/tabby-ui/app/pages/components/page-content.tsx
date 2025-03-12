'use client'

import { useContext, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { MessageMarkdown } from '@/components/message-markdown'

import { PageItem } from '../types'
import { MessageContentForm } from './message-content-form'
import { PageContext } from './page-context'

const updatePageContentMutation = graphql(/* GraphQL */ `
  mutation updatePageContent($input: UpdatePageContentInput!) {
    updatePageContent(input: $input)
  }
`)

interface Props {
  page: PageItem | undefined
  onUpdate: (content: string) => void
}

export function PageContent({ page, onUpdate }: Props) {
  const { mode, isLoading, isPageOwner } = useContext(PageContext)
  const [showForm, setShowForm] = useState(false)

  const updatePageContent = useMutation(updatePageContentMutation)

  const handleSubmitContentChange = async (content: string) => {
    if (!page) return

    const result = await updatePageContent({
      input: {
        id: page.id,
        content
      }
    })

    if (result?.data?.updatePageContent) {
      onUpdate(content)
      setShowForm(false)
    } else {
      let error = result?.error
      return error
    }
  }

  return (
    <div>
      {showForm ? (
        <MessageContentForm
          message={page?.content ?? ''}
          onCancel={() => setShowForm(false)}
          onSubmit={handleSubmitContentChange}
        />
      ) : (
        <MessageMarkdown
          message={page?.content ?? ''}
          supportsOnApplyInEditorV2={false}
        />
      )}
      {isPageOwner && mode === 'edit' && !isLoading && !showForm && (
        <div className="mt-3">
          <Button
            size="sm"
            variant="ghost"
            className="h-auto gap-0.5 px-2 py-1 font-medium text-foreground/60"
            disabled={isLoading}
            onClick={() => {
              setShowForm(true)
            }}
          >
            <IconEdit />
            Edit
          </Button>
        </div>
      )}
    </div>
  )
}
