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

/**
 * Retrieves the name of the completion query from a given string@.
 * @param {string} val - The input string to search for the completion query name.
 * @param {number | undefined} selectionEnd - The index at which the selection ends in the input string.
 * @return {string | undefined} - The name of the completion query if found, otherwise undefined.
 */
export function getSearchCompletionQueryName(
  val: string,
  selectionEnd: number | undefined
): string | undefined {
  const queryString = val.substring(0, selectionEnd)
  const matches = /@(\w+)$/.exec(queryString)
  return matches?.[1]
}
