import React, { useState } from 'react'

import { useStore } from '@/lib/hooks/use-store'
import { useChatStore } from '@/lib/stores/chat-store'
import { cn } from '@/lib/utils'
import fetcher from '@/lib/tabby/fetcher'
import { Button } from '@/components/ui/button'
import { IconClose } from '@/components/ui/icons'

import { QuickActionEventPayload } from '../lib/event-emitter'
import { SourceCodeBrowserContext } from './source-code-browser'
import { ISearchHit, SearchReponse } from '@/lib/types'

interface ChatSideBarProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'children'> { }

export const ChatSideBar: React.FC<ChatSideBarProps> = ({
  className,
  ...props
}) => {
  const { pendingEvent, setPendingEvent } = React.useContext(
    SourceCodeBrowserContext
  )
  const activeChatId = useStore(useChatStore, state => state.activeChatId)
  const iframeRef = React.useRef<HTMLIFrameElement>(null)

  const getPrompt = async ({
    action,
    code,
    language,
    path,
    lineFrom,
    lineTo
  }: QuickActionEventPayload) => {
    const contextPrompt = await buildContextPrompt(language, code, path);
    let builtInPrompt = ''
    switch (action) {
      case 'explain':
        builtInPrompt = 'Explain the following code:'
        break
      case 'generate_unittest':
        builtInPrompt = 'Generate a unit test for the following code:'
        break
      case 'generate_doc':
        builtInPrompt = 'Generate documentation for the following code:'
        break
      default:
        break
    }
    const codeBlockMeta = `${language ?? ''
      } is_reference=1 path=${path} line_from=${lineFrom} line_to=${lineTo}`;
    return `${contextPrompt}${builtInPrompt}\n${'```'}${codeBlockMeta}\n${code}\n${'```'}\n`
  }

  async function postPrompt(e: QuickActionEventPayload) {
    const contentWindow = iframeRef.current?.contentWindow
    contentWindow?.postMessage({
      action: 'append',
      payload: await getPrompt(e)
    })
  }

  React.useEffect(() => {
    if (pendingEvent) {
      postPrompt(pendingEvent).then(() => {
        setPendingEvent(undefined)
      })
    }
  }, [pendingEvent, iframeRef.current?.contentWindow])

  return (
    <div className={cn('flex h-full flex-col', className)} {...props}>
      <Header />
      <iframe
        src={`/playground`}
        className="w-full flex-1 border-0"
        key={activeChatId}
        ref={iframeRef}
      />
    </div>
  )
}

function Header() {
  const { setChatSideBarVisible } = React.useContext(SourceCodeBrowserContext)

  return (
    <div className="sticky top-0 flex items-center justify-end px-2 py-1">
      <Button
        size="icon"
        variant="ghost"
        onClick={e => setChatSideBarVisible(false)}
      >
        <IconClose />
      </Button>
    </div>
  )
}

async function buildContextPrompt(language: string | undefined, code: string, path: string | undefined) {
  if (!language || !path) {
    return [];
  }

  if (code.length < 128) {
    return [];
  }

  const repo = path = path.split("/")[0];

  const tokens = code.split(/[^\w]/).filter(x => x);

  // FIXME(meng): restrict query with `git_url` of `repo`.
  const languageQuery = buildLanguageQuery(language)
  const bodyQuery = tokens.map(x => `body:${x}`).join(' OR ');
  const query = `${languageQuery} AND (${bodyQuery})`

  const queryParam = `q=${encodeURIComponent(query)}&limit=20`;

  const data: SearchReponse = await fetcher(`/v1beta/search?${queryParam}`, {
    responseFormat: "json"
  });
  const snippets = data.hits.filter(x => x.score > 30 && path !== x.doc.filepath) || [];
  return formatContextPrompt(repo, language, snippets.slice(0, 3));
}

function formatContextPrompt(repo: string, language: string, snippets: ISearchHit[]) {
  let prompt = "Given following relevant code snippets:\n\n";
  for (const { doc } of snippets) {
    const numLines = doc.body.split(/\r\n|\r|\n/).length;
    const fromLine = doc.start_line;
    const toLine = doc.start_line + numLines - 1;
    const reference = `\`\`\`${language} is_reference=1 path=${repo}/${doc.filepath} line_from=${fromLine} line_to=${toLine}
${doc.body}
\`\`\`
`;
    prompt += reference
  }

  if (snippets.length) {
    return prompt;
  } else {
    return '';
  }
}

function buildLanguageQuery(language: string) {
  if (language == "javascript" || language == "jsx" || language == "typescript" || language == "tsx") {
    language = "javascript-typescript"
  }

  return `language:${language}`;
}