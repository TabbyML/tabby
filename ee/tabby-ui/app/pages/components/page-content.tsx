'use client'

import { useContext, useState } from 'react'

import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { MessageMarkdown } from '@/components/message-markdown'

import { PageItem } from '../types'
import { MessageContentForm } from './message-content-form'
import { PageContext } from './page-context'

interface Props {
  page: PageItem | undefined
}

export function PageContent({ page }: Props) {
  const { mode, isLoading, isPageOwner } = useContext(PageContext)
  const [showForm, setShowForm] = useState(false)

  const handleSubmitContentChange = async (message: string) => {
    // todo submit change
    setShowForm(false)
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
            variant="hover-destructive"
            className="h-auto gap-0.5 px-2 py-1 font-normal"
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
