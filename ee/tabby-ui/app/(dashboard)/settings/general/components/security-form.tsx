'use client'

import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { compact, isEmpty } from 'lodash-es'
import { useFieldArray, useForm } from 'react-hook-form'
import * as z from 'zod'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconTrash } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'

const formSchema = z.object({
  disableTelemetry: z.boolean(),
  // https://github.com/shadcn-ui/ui/issues/384
  // https://github.com/shadcn-ui/ui/blob/main/apps/www/app/examples/forms/profile-form.tsx
  domainList: z
    .array(
      z.object({
        value: z.string()
      })
    )
    .optional()
})

type SecurityFormValues = z.infer<typeof formSchema>

interface SecurityFormProps {
  defaultValues?: Omit<Partial<SecurityFormValues>, 'domainList'> & {
    domainList?: string[]
  }
  onSuccess?: () => void
}

export const GeneralSecurityForm: React.FC<SecurityFormProps> = ({
  onSuccess,
  defaultValues: propsDefaultValues
}) => {
  const defaultValues = React.useMemo(() => {
    const _defaultValues: SecurityFormProps['defaultValues'] =
      propsDefaultValues ?? {}
    const { domainList, ...values } = _defaultValues
    return {
      disableTelemetry: false,
      ...values,
      domainList: buildListFieldFromValues(domainList) ?? [{ value: '' }]
    }
  }, [propsDefaultValues])

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues
  })

  const { fields, append, remove, update } = useFieldArray({
    control: form.control,
    name: 'domainList'
  })

  const isDirty = !isEmpty(form.formState.dirtyFields)

  const onRemoveDomainItem = (index: number) => {
    if (fields?.length === 1 && index === 0) {
      update(index, { value: '' })
    } else {
      remove(index)
    }
  }

  const handleDomainListKeyDown: React.KeyboardEventHandler<
    HTMLInputElement
  > = e => {
    if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
      e.preventDefault()
      append({ value: '' })
    }
  }

  const onSubmit = ({ domainList, ...values }: SecurityFormValues) => {
    const formattedValues = {
      // format domainList before submit, Array<{ value: string }> => Array<string>
      domainList: buildListValuesFromField(domainList),
      ...values
    }
    // todo submit values
  }

  return (
    <Form {...form}>
      <div className="flex flex-col gap-4">
        <form
          className="flex flex-col gap-8"
          onSubmit={form.handleSubmit(onSubmit)}
        >
          <FormField
            control={form.control}
            name="disableTelemetry"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center gap-1">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <FormLabel className="cursor-pointer">
                    Disabling Client Side Telemetry
                  </FormLabel>
                </div>
                <FormDescription>
                  When activated, the client-side telemetry (IDE/Extensions)
                  will be disabled, regardless of the client-side settings.
                </FormDescription>
              </FormItem>
            )}
          />
          <div>
            {fields.map((field, index) => (
              <FormField
                control={form.control}
                key={field.id}
                name={`domainList.${index}.value`}
                render={({ field: itemField }) => (
                  <FormItem>
                    <FormLabel className={cn(index !== 0 && 'sr-only')}>
                      Domain List for Register (without an Invitation)
                    </FormLabel>
                    <FormDescription className={cn(index !== 0 && 'sr-only')}>
                      Add domains for register without an invitation.
                    </FormDescription>
                    <div className="flex items-center gap-2">
                      <FormControl>
                        <Input
                          placeholder="e.g. tabbyml.com"
                          {...itemField}
                          onKeyDown={handleDomainListKeyDown}
                        />
                      </FormControl>
                      <Button
                        variant="hover-destructive"
                        onClick={e => onRemoveDomainItem(index)}
                      >
                        <IconTrash />
                      </Button>
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
            ))}
            <div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => append({ value: '' })}
              >
                Add domain
              </Button>
            </div>
          </div>
          <div className="mt-2 flex justify-end">
            <Button type="submit" disabled={!isDirty}>
              Update
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}

function buildListFieldFromValues(list: string[] | undefined) {
  return list?.map(item => ({ value: item }))
}

function buildListValuesFromField(fieldListValue?: Array<{ value: string }>) {
  const list = compact(fieldListValue?.map(item => item.value))
  return list.length ? list : undefined
}
