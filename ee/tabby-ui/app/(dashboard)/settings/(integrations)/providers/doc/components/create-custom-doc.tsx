'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { graphql } from '@/lib/gql/generates'
import { useMutation } from '@/lib/tabby/gql'
import { Button } from '@/components/ui/button'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'

const createCustomDocumentMutation = graphql(/* GraphQL */ `
  mutation CreateCustomDocument($input: CreateCustomDocumentInput!) {
    createCustomDocument(input: $input)
  }
`)

const formSchema = z.object({
  name: z.string().trim(),
  url: z.string().url().trim()
})

type FormValues = z.infer<typeof formSchema>

export default function CreateCustomDocument() {
  const router = useRouter()
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const onCreated = () => {
    router.push('./')
  }

  const { isSubmitting } = form.formState
  const createCustomDocument = useMutation(createCustomDocumentMutation, {
    onCompleted() {
      form.reset({ url: undefined })
      onCreated()
    },
    form
  })

  const onSubmit = (values: FormValues) => {
    return createCustomDocument({
      input: values
    })
  }

  return (
    <>
      <Form {...form}>
        <div className="grid gap-2">
          <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>Name</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. tabby"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="url"
              render={({ field }) => (
                <FormItem>
                  <FormLabel required>URL</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="e.g. https://tabby.tabbyml.com/"
                      autoCapitalize="none"
                      autoCorrect="off"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="flex justify-end gap-4">
              <Button
                type="button"
                variant="ghost"
                disabled={isSubmitting}
                onClick={() => router.back()}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                Create
              </Button>
            </div>
          </form>
          <FormMessage className="text-center" />
        </div>
      </Form>
    </>
  )
}
