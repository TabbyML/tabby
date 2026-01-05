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
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { TagInput } from '@/components/ui/tag-input'

const createRepositoryMutation = graphql(/* GraphQL */ `
  mutation createGitRepository(
    $name: String!
    $gitUrl: String!
    $refs: [String!]
  ) {
    createGitRepository(name: $name, gitUrl: $gitUrl, refs: $refs)
  }
`)

const formSchema = z.object({
  name: z.string(),
  gitUrl: z.string(),
  refs: z.array(z.string()).optional()
})

export default function CreateRepositoryForm({
  onCreated
}: {
  onCreated: () => void
}) {
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const createRepository = useMutation(createRepositoryMutation, {
    onCompleted() {
      form.reset({ name: undefined, gitUrl: undefined, refs: undefined })
      onCreated()
    },
    form
  })

  const onSubmit = (values: z.infer<typeof formSchema>) => {
    createRepository({
      name: values.name,
      gitUrl: values.gitUrl,
      refs: values.refs && values.refs.length > 0 ? values.refs : undefined
    })
  }

  const router = useRouter()
  return (
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
                    autoComplete="off"
                    {...field}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={form.control}
            name="gitUrl"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Git URL</FormLabel>
                <FormDescription>Remote or local Git URL</FormDescription>
                <FormControl>
                  <Input
                    placeholder="e.g. https://github.com/TabbyML/tabby"
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
            name="refs"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Branches</FormLabel>
                <FormDescription>
                  Branches to index (press Enter to select, leave empty for
                  default branch)
                </FormDescription>
                <FormControl>
                  <TagInput
                    placeholder="e.g. main"
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
  )
}
