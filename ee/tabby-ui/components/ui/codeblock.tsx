// Inspired by Chatbot-UI and modified to fit the needs of this project
// @see https://github.com/mckaywrigley/chatbot-ui/blob/main/components/Markdown/CodeBlock.tsx

'use client'

import { FC, memo, useState } from 'react'
import {
  createElement,
  Prism as SyntaxHighlighter
} from 'react-syntax-highlighter'
import { coldarkDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'

import { useCopyToClipboard } from '@/lib/hooks/use-copy-to-clipboard'
import { Button } from '@/components/ui/button'
import {
  IconAlignJustify,
  IconApplyInEditor,
  IconCheck,
  IconCopy,
  IconWrapText
} from '@/components/ui/icons'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger
} from '@/components/ui/tooltip'

export interface CodeBlockProps {
  language: string
  value: string
  onCopyContent?: (value: string) => void
  onApplyInEditor?: (value: string) => void
  canWrapLongLines?: boolean
}

interface languageMap {
  [key: string]: string | undefined
}

export const programmingLanguages: languageMap = {
  javascript: '.js',
  python: '.py',
  java: '.java',
  c: '.c',
  cpp: '.cpp',
  'c++': '.cpp',
  'c#': '.cs',
  ruby: '.rb',
  php: '.php',
  swift: '.swift',
  'objective-c': '.m',
  kotlin: '.kt',
  typescript: '.ts',
  go: '.go',
  perl: '.pl',
  rust: '.rs',
  scala: '.scala',
  haskell: '.hs',
  lua: '.lua',
  shell: '.sh',
  sql: '.sql',
  html: '.html',
  css: '.css'
  // add more file extensions here, make sure the key is same as language prop in CodeBlock.tsx component
}

export const generateRandomString = (length: number, lowercase = false) => {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXY3456789' // excluding similar looking characters like Z, 2, I, 1, O, 0
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return lowercase ? result.toLowerCase() : result
}

const CodeBlock: FC<CodeBlockProps> = memo(
  ({ language, value, onCopyContent, onApplyInEditor, canWrapLongLines }) => {
    const [wrapLongLines, setWrapLongLines] = useState(false)
    const { isCopied, copyToClipboard } = useCopyToClipboard({
      timeout: 2000,
      onCopyContent
    })

    const onCopy = () => {
      if (isCopied) return
      copyToClipboard(value)
    }

    // react-syntax-highlighter does not render .toml files correctly
    // using bash syntax as a workaround for better display
    const languageForSyntax = language === 'toml' ? 'bash' : language
    return (
      <div className="codeblock relative w-full bg-zinc-950 font-sans">
        <div className="flex w-full items-center justify-between bg-zinc-800 px-6 py-2 pr-4 text-zinc-100">
          <span className="text-xs lowercase">{language}</span>
          <div className="flex items-center space-x-1">
            {canWrapLongLines && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="text-xs hover:bg-[#3C382F] hover:text-[#F4F4F5] focus-visible:ring-1 focus-visible:ring-slate-700 focus-visible:ring-offset-0"
                    onClick={() => setWrapLongLines(!wrapLongLines)}
                  >
                    {wrapLongLines ? <IconAlignJustify /> : <IconWrapText />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="m-0">Toggle word wrap</p>
                </TooltipContent>
              </Tooltip>
            )}
            {onApplyInEditor && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-xs hover:bg-[#3C382F] hover:text-[#F4F4F5] focus-visible:ring-1 focus-visible:ring-slate-700 focus-visible:ring-offset-0"
                    onClick={() => onApplyInEditor(value)}
                  >
                    <IconApplyInEditor />
                    <span className="sr-only">Apply in Editor</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="m-0">Apply in Editor</p>
                </TooltipContent>
              </Tooltip>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-xs hover:bg-[#3C382F] hover:text-[#F4F4F5] focus-visible:ring-1 focus-visible:ring-slate-700 focus-visible:ring-offset-0"
                  onClick={onCopy}
                >
                  {isCopied ? <IconCheck /> : <IconCopy />}
                  <span className="sr-only">Copy</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p className="m-0">Copy</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </div>
        <SyntaxHighlighter
          language={languageForSyntax}
          style={coldarkDark}
          PreTag="div"
          showLineNumbers
          wrapLongLines={wrapLongLines}
          customStyle={{
            margin: 0,
            width: '100%',
            background: 'transparent',
            padding: '1.5rem 1rem'
          }}
          codeTagProps={{
            style: {
              fontSize: '0.9rem',
              fontFamily: 'var(--font-mono)'
            }
          }}
          renderer={({ rows, stylesheet, useInlineStyles }) => {
            return rows.map((row, index) => {
              const children = row.children
              const lineNumberElement = children?.shift()

              /**
               * We will take current structure of the rows and rebuild it
               * according to the suggestion here https://github.com/react-syntax-highlighter/react-syntax-highlighter/issues/376#issuecomment-1246115899
               */
              if (lineNumberElement) {
                row.children = [
                  lineNumberElement,
                  {
                    children,
                    properties: {
                      className: []
                    },
                    tagName: 'span',
                    type: 'element'
                  }
                ]
              }

              return createElement({
                node: row,
                stylesheet,
                useInlineStyles,
                key: index
              })
            })
          }}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    )
  }
)
CodeBlock.displayName = 'CodeBlock'

export { CodeBlock }
