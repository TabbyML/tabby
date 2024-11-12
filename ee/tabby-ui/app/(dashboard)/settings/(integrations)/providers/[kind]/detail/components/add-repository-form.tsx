'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

import {
  IntegrationKind,
  IntegrationStatus,
  ListIntegratedRepositoriesQuery
} from '@/lib/gql/generates/graphql'
import { useMutation } from '@/lib/tabby/gql'
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
  IconChevronUpDown,
  IconSpinner
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'

import { updateIntegratedRepositoryActiveMutation } from '../query'

const formSchema = z.object({
  id: z.string()
})

type ActivateRepositoryFormValues = z.infer<typeof formSchema>

interface ActivateRepositoryFormProps {
  kind: IntegrationKind
  onCreated?: (id: string) => void
  onCancel: () => void
  providerStatus: IntegrationStatus | undefined
  repositories:
    | ListIntegratedRepositoriesQuery['integratedRepositories']['edges']
    | undefined
  fetchingRepos: boolean
}

export default function AddRepositoryForm({
  onCreated,
  onCancel,
  repositories,
  providerStatus,
  fetchingRepos
}: ActivateRepositoryFormProps) {
  const [open, setOpen] = React.useState(false)
  const form = useForm<ActivateRepositoryFormValues>({
    resolver: zodResolver(formSchema)
  })
  const commandListRef = React.useRef<HTMLDivElement>(null)

  const { isSubmitting } = form.formState

  const emptyText = React.useMemo(() => {
    switch (providerStatus) {
      case IntegrationStatus.Pending:
        return 'Awaiting the next data synchronization'
      case IntegrationStatus.Failed:
        return 'Synchronizing error. Please check if the access token is still valid'
      default:
        return 'No repository found'
    }
  }, [providerStatus])

  const updateProvidedRepositoryActive = useMutation(
    updateIntegratedRepositoryActiveMutation,
    {
      form
    }
  )

  const onSubmit = (values: ActivateRepositoryFormValues) => {
    const id = values.id

    return updateProvidedRepositoryActive({
      id: values.id,
      active: true
    }).then(res => {
      if (res?.data?.updateIntegratedRepositoryActive) {
        form.reset({ id: undefined })
        onCreated?.(id)
      }
    })
  }

  const scrollCommandListToTop = () => {
    requestAnimationFrame(() => {
      if (commandListRef.current) {
        commandListRef.current.scrollTop = 0
      }
    })
  }

  const onSearchChange = () => {
    scrollCommandListToTop()
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
                      <CommandInput
                        placeholder="Search repository..."
                        onValueChange={onSearchChange}
                      />
                      <CommandList
                        className="max-h-[30vh]"
                        ref={commandListRef}
                      >
                        <CommandEmpty>
                          {fetchingRepos ? (
                            <div className="flex justify-center">
                              <IconSpinner className="h-6 w-6" />
                            </div>
                          ) : (
                            emptyText
                          )}
                        </CommandEmpty>
                        <CommandGroup>
                          {providerStatus !== IntegrationStatus.Pending &&
                            repositories?.map(repo => (
                              <CommandItem
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
