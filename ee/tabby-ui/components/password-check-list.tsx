import React from 'react'
import { isNil } from 'lodash-es'
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
  .refine(password => password.length >= 8, {
    params: { errorCode: PASSWORD_ERRORCODE.AT_LEAST_EIGHT_CHAR }
  })
  .refine(password => password.length <= 20, {
    params: { errorCode: PASSWORD_ERRORCODE.AT_MOST_TWENTY_CHAT }
  })

export const usePasswordErrors = (password?: string) => {
  const [passworErrors, setPasswordErrors] = React.useState<
    PASSWORD_ERRORCODE[]
  >([])

  React.useEffect(() => {
    if (!isNil(password)) {
      try {
        passwordSchema.parse(password)
        setPasswordErrors([])
      } catch (err) {
        if (err instanceof z.ZodError) {
          setPasswordErrors(
            err.issues.map((error: any) => error.params.errorCode)
          )
        }
      }
    }
  }, [password])

  return [passworErrors, setPasswordErrors]
}

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
  function Rule({
    errorCode,
    text
  }: {
    errorCode: PASSWORD_ERRORCODE
    text: string
  }) {
    return (
      <li
        className={cn('py-0.5', {
          'text-green-600 dark:text-green-500':
            password.length > 0 && !passworErrors.includes(errorCode),
          'text-red-600 dark:text-red-500':
            showPasswordError &&
            password.length > 0 &&
            passworErrors.includes(errorCode)
        })}
      >
        {text}
      </li>
    )
  }

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
        <Rule
          errorCode={PASSWORD_ERRORCODE.AT_LEAST_EIGHT_CHAR}
          text="At least 8 characters long"
        />
        <Rule
          errorCode={PASSWORD_ERRORCODE.AT_MOST_TWENTY_CHAT}
          text="No more than 20 characters long"
        />
        <Rule
          errorCode={PASSWORD_ERRORCODE.LOWERCASE_MSISSING}
          text="At least one lowercase character"
        />
        <Rule
          errorCode={PASSWORD_ERRORCODE.UPPERCASE_MSISSING}
          text="At least one uppercase character"
        />
        <Rule
          errorCode={PASSWORD_ERRORCODE.NUMBER_MISSING}
          text="At least one numeric character"
        />
        <Rule
          errorCode={PASSWORD_ERRORCODE.SPECIAL_CHAR_MISSING}
          text={`At least one special character , such as @#$%^&{}`}
        />
      </ul>
    </div>
  )
}
