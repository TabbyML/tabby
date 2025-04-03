import { useContext, useState } from 'react'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'

import { PageItem } from '../types'
import { MessageContentForm } from './message-content-form'
import { PageContext } from './page-context'

const updatePageTitleMutation = graphql(/* GraphQL */ `
  mutation updatePageTitle($input: UpdatePageTitleInput!) {
    updatePageTitle(input: $input)
  }
`)

interface Props {
  page: PageItem | undefined
  onUpdate: (content: string) => void
  isGeneratingPageTitle: boolean
}

export function PageTitle({ page, onUpdate, isGeneratingPageTitle }: Props) {
  const { mode, isLoading } = useContext(PageContext)
  const [showForm, setShowForm] = useState(false)

  const updatePageTitle = useMutation(updatePageTitleMutation)

  const handleSubmitTitleChange = async (title: string) => {
    if (!page) return

    const result = await updatePageTitle({
      input: {
        id: page.id,
        title
      }
    })

    if (result?.data?.updatePageTitle) {
      onUpdate(title)
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
          message={page?.title ?? ''}
          onSubmit={handleSubmitTitleChange}
          onCancel={() => setShowForm(false)}
          inputClassName="text-3xl font-semibold"
        />
      ) : (
        <div className="flex gap-2">
          <h1
            className={cn('text-4xl font-semibold', {
              'animate-pulse text-muted-foreground': isGeneratingPageTitle
            })}
          >
            {page?.title}
          </h1>
          {mode === 'edit' && !isLoading && (
            <Button
              size="icon"
              className="mt-1 shrink-0 rounded-full shadow-none"
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
