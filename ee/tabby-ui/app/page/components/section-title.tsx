import { HTMLAttributes, useContext } from 'react'

import { Section } from '@/lib/gql/generates/graphql'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { IconEdit } from '@/components/ui/icons'
import { MessageMarkdown } from '@/components/message-markdown'

import { PageContext } from './page-context'

interface QuestionBlockProps extends HTMLAttributes<HTMLDivElement> {
  message: Section
}

export function SectionTitle({
  message,
  className,
  ...props
}: QuestionBlockProps) {
  const { mode } = useContext(PageContext)
  return (
    <div>
      <div
        className={cn('flex items-center gap-2 font-semibold', className)}
        id={message.id}
        {...props}
      >
        {/* FIXME: use markdown? */}
        <MessageMarkdown
          message={message.title}
          contextInfo={undefined}
          supportsOnApplyInEditorV2={false}
          className="text-3xl prose-h2:text-foreground prose-p:mb-1 prose-p:mt-0"
          headline
        />
        {mode === 'edit' && (
          <Button variant="outline" className="px-2">
            <IconEdit className="h-6 w-6" />
          </Button>
        )}
      </div>
      {/* mock tags */}
      {/* <div className="mb-4 mt-1 flex items-center gap-2">
        <Badge variant="secondary">
          <IconGithub className="mr-1" />
          TabbyML/tabby
        </Badge>
        <Badge variant="secondary">
          <IconEmojiBook className="mr-1" />
          Tailwindcss
        </Badge>
        <Badge variant="secondary">
          <IconEmojiBook className="mr-1" />
          Pundit
        </Badge>
      </div> */}
    </div>
  )
}
