// Minor change for contribution test
import { Fragment, ReactNode, useContext, useMemo, useState } from 'react'
import { compact, flatten, isNil } from 'lodash-es'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import {
  ContextInfo,
  Maybe,
  MessageAttachmentClientCode
} from '@/lib/gql/generates/graphql'
import {
  AttachmentCodeItem,
  AttachmentDocItem,
  Context,
  RelevantCodeContext
} from '@/lib/types'
import {
  cn,
  convertFromFilepath,
  convertToFilepath,
  encodeMentionPlaceHolder,
  formatCustomHTMLBlockTags,
  getRangeFromAttachmentCode,
  isAttachmentCommitDoc,
  isAttachmentIngestedDoc,
  resolveDirectoryPath,
  resolveFileNameForDisplay
} from '@/lib/utils'
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger
} from '@/components/ui/hover-card'
import { MemoizedReactMarkdown } from '@/components/markdown'
import './style.css'
import { FileBox, SquareFunctionIcon } from 'lucide-react'
import {
  FileLocation,
  Filepath,
  ListSymbolItem,
  LookupSymbolHint,
  SymbolInfo
} from 'tabby-chat-panel/index'
import {
  CUSTOM_HTML_BLOCK_TAGS,
  CUSTOM_HTML_INLINE_TAGS
} from '@/lib/constants'
import {
  MARKDOWN_CITATION_REGEX,
  MARKDOWN_COMMAND_REGEX,
  MARKDOWN_FILE_REGEX,
  MARKDOWN_SOURCE_REGEX,
  MARKDOWN_SYMBOL_REGEX
} from '@/lib/constants/regex'
import { Mention } from '../mention-tag'
import { IconFile, IconFileText } from '../ui/icons'
import { Skeleton } from '../ui/skeleton'
import { CodeElement } from './code'
import { customStripTagsPlugin } from './custom-strip-tags-plugin'
import { DocDetailView } from './doc-detail-view'
import { MessageMarkdownContext } from './markdown-context'
type RelevantDocItem = {
