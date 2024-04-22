import * as z from 'zod'

import { cn } from '@/lib/utils'

export enum PASSWORD_ERRORCODE {
  LOWERCASE_MSISSING = 'lowercase_missing',
  UPPERCASE_MSISSING = 'uppercase_missing',
  NUMBER_MISSING = 'number_missing',
  SPECIAL_CHAR_MISSING = 'special_char_missing',
  AT_LEAST_EIGHT_CHAR = 'at_least_eight_char',
  AT_MOST_TWENTY_CHAT = 'at_most_twenty_char'
}

export const passwordSchema = z
  .string()
  .refine(password => /[a-z]/.test(password), {
    params: { errorCode: PASSWORD_ERRORCODE.LOWERCASE_MSISSING }
  })
  .refine(password => /[A-Z]/.test(password), {
    params: { errorCode: PASSWORD_ERRORCODE.UPPERCASE_MSISSING }
  })
  .refine(password => /\d/.test(password), {
    params: { errorCode: PASSWORD_ERRORCODE.NUMBER_MISSING }
  })
  .refine(password => /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+/.test(password), {
    params: { errorCode: PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING }
  })
  .refine(password => password.length > 8, {
    params: { errorCode: PASSWORD_ERRORCODE.AT_LEAST_EIGHT_CHAR }
  })
  .refine(password => password.length < 20, {
    params: { errorCode: PASSWORD_ERRORCODE.AT_MOST_TWENTY_CHAT }
  })

export function PasswordCheckList({
  password,
  showPasswordSchema,
  passworErrors,
  showPasswordError
}: {
  password: string
  showPasswordSchema: boolean
  passworErrors: PASSWORD_ERRORCODE[]
  showPasswordError: boolean
}) {
  return (
    <div
      className={cn('relative text-sm transition-all', {
        'h-0 opacity-0 -z-10': !showPasswordSchema,
        'mt-4 h-40 opacity-100': showPasswordSchema
      })}
    >
      <p className="mb-0.5 text-xs text-muted-foreground">
        Set up a strong password with
      </p>
      <ul className="list-disc pl-4">
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.AT_LEAST_EIGHT_CHAR),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.AT_LEAST_EIGHT_CHAR)
          })}
        >
          At least 8 characters long
        </li>
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.AT_MOST_TWENTY_CHAT),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.AT_MOST_TWENTY_CHAT)
          })}
        >
          No more than 20 characters long
        </li>
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.LOWERCASE_MSISSING),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.LOWERCASE_MSISSING)
          })}
        >
          At least one lowercase character
        </li>
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.UPPERCASE_MSISSING),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.UPPERCASE_MSISSING)
          })}
        >
          At least one uppercase character
        </li>
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.NUMBER_MISSING),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.NUMBER_MISSING)
          })}
        >
          At least one numeric character
        </li>
        <li
          className={cn('py-0.5', {
            'text-green-600':
              password.length > 0 &&
              !passworErrors.includes(PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING),
            'text-red-600':
              showPasswordError &&
              password.length > 0 &&
              passworErrors.includes(PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING)
          })}
        >{`At least one special character , such as @#$%^&{}`}</li>
      </ul>
    </div>
  )
}
