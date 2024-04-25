'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import { QueryResponseData, useMutation } from '@/lib/tabby/gql'
import { listGithubRepositories } from '@/lib/tabby/query'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList
} from '@/components/ui/command'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage
} from '@/components/ui/form'
import {
  IconCheck,
  IconChevronUpDown
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'

import { updateGithubProvidedRepositoryActiveMutation } from '../query'

const formSchema = z.object({
  id: z.string()
})

type LinkRepositoryFormValues = z.infer<typeof formSchema>

export default function LinkRepositoryForm({
  onCreated,
  onCancel,
  repositories
}: {
  onCreated?: () => void
  onCancel: () => void
  repositories:
    | QueryResponseData<
        typeof listGithubRepositories
      >['githubRepositories']['edges']
    | undefined
}) {
  const [open, setOpen] = React.useState(false)
  const form = useForm<LinkRepositoryFormValues>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  const updateGithubProvidedRepositoryActive = useMutation(
    updateGithubProvidedRepositoryActiveMutation,
    {
      onCompleted(data) {
        if (data?.updateGithubProvidedRepositoryActive) {
          form.reset({ id: undefined })
          onCreated?.()
        }
      },
      form
    }
  )

  const onSubmit = (values: LinkRepositoryFormValues) => {
    return updateGithubProvidedRepositoryActive({
      id: values.id,
      active: true
    })
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="id"
            render={({ field }) => (
              <FormItem className="flex flex-col">
                <Popover open={open} onOpenChange={setOpen}>
                  <PopoverTrigger asChild>
                    <FormControl>
                      <Button
                        variant="outline"
                        role="combobox"
                        className={cn(
                          'justify-between font-normal',
                          !field.value && 'text-muted-foreground'
                        )}
                      >
                        {field.value
                          ? repositories?.find(
                              repo => repo.node.id === field.value
                            )?.node?.gitUrl
                          : 'Select repository'}
                        <IconChevronUpDown />
                      </Button>
                    </FormControl>
                  </PopoverTrigger>
                  <PopoverContent
                    className="w-[var(--radix-popover-trigger-width)] p-0"
                    align="start"
                    side="bottom"
                  >
                    <Command>
                      <CommandInput placeholder="Search repository..." />
                      <CommandList className="max-h-60">
                        <CommandEmpty>No repository found.</CommandEmpty>
                        <CommandGroup>
                          {repositories?.map(repo => (
                            <CommandItem
                              // value={repo.node.id}
                              key={repo.node.id}
                              onSelect={() => {
                                form.setValue('id', repo.node.id)
                                setOpen(false)
                              }}
                            >
                              <IconCheck
                                className={cn(
                                  'mr-2',
                                  repo.node.id === field.value
                                    ? 'opacity-100'
                                    : 'opacity-0'
                                )}
                              />
                              {repo.node.gitUrl}
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
                <FormMessage />
              </FormItem>
            )}
          />
          <div className="flex justify-end gap-4">
            <Button
              type="button"
              variant="ghost"
              disabled={isSubmitting}
              onClick={onCancel}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              Add
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}
