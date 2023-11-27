import { clsx, type ClassValue } from 'clsx'
import { customAlphabet } from 'nanoid'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const nanoid = customAlphabet(
  '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
  7
) // 7-character random string

export async function fetcher<JSON = any>(
  input: RequestInfo,
  init?: RequestInit
): Promise<JSON> {
  const res = await fetch(input, init)

  if (!res.ok) {
    const json = await res.json()
    if (json.error) {
      const error = new Error(json.error) as Error & {
        status: number
      }
      error.status = res.status
      throw error
    } else {
      throw new Error('An unexpected error occurred')
    }
  }

  return res.json()
}

export function formatDate(input: string | number | Date): string {
  const date = new Date(input)
  return date.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric'
  })
}

export function truncateText(
  text: string,
  maxLength = 50,
  delimiters = /[ ,.:;\n，。：；]/
) {
  if (!text) return ''
  if (text.length <= maxLength) {
    return text
  }

  let truncatedText = text.slice(0, maxLength)

  let lastDelimiterIndex = -1
  for (let i = maxLength - 1; i >= 0; i--) {
    if (delimiters.test(truncatedText[i])) {
      lastDelimiterIndex = i
      break
    }
  }

  if (lastDelimiterIndex !== -1) {
    truncatedText = truncatedText.slice(0, lastDelimiterIndex)
  }

  return truncatedText + '...'
}
