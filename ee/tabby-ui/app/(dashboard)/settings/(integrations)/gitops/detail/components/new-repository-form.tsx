'use client'

import * as React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'

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
  FormLabel,
  FormMessage
} from '@/components/ui/form'
import { IconCheck, IconChevronUpDown } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'

const formSchema = z.object({
  gitUrl: z.string(),
  name: z.string()
})

type LinkRepositoryFormValues = z.infer<typeof formSchema>

const repos = [
  {
    value: 'https://github.com/tabbyml/tabby',
    label: 'https://github.com/tabbyml/tabby'
  },
  {
    value: 'https://github.com/tabbyml/test',
    label: 'https://github.com/tabbyml/test'
  }
]

export default function LinkRepositoryForm({
  onCreated,
  onCancel
}: {
  onCreated?: (values: LinkRepositoryFormValues) => void
  onCancel: () => void
}) {
  const [open, setOpen] = React.useState(false)
  const form = useForm<LinkRepositoryFormValues>({
    resolver: zodResolver(formSchema)
  })

  const { isSubmitting } = form.formState
  // const createRepository = useMutation(createRepositoryMutation, {
  //   onCompleted() {
  //     form.reset({ name: undefined, gitUrl: undefined })
  //     onCreated()
  //   },
  //   form
  // })

  const git_url = form.watch('gitUrl')

  React.useEffect(() => {
    if (!git_url) {
      form.setValue('name', '')
    } else {
      const name = resolveRepoNameFromUrl(git_url) || git_url
      form.setValue('name', name)
    }
  }, [git_url])

  const onSubmit = (values: LinkRepositoryFormValues) => {
    onCreated?.(values)
  }

  return (
    <Form {...form}>
      <div className="grid gap-2">
        <form className="grid gap-6" onSubmit={form.handleSubmit(onSubmit)}>
          <FormField
            control={form.control}
            name="gitUrl"
            render={({ field }) => (
              <FormItem className="flex flex-col">
                <FormLabel required>Repository</FormLabel>
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
                          ? repos.find(repo => repo.value === field.value)
                              ?.label
                          : 'Select repository'}
                        <IconChevronUpDown />
                      </Button>
                    </FormControl>
                  </PopoverTrigger>
                  <PopoverContent
                    className="p-0 w-[var(--radix-popover-trigger-width)]"
                    align="start"
                  >
                    <Command>
                      <CommandInput placeholder="Search repository..." />
                      <CommandList>
                        {/* todo loading */}
                        <CommandEmpty>No repository found.</CommandEmpty>
                        <CommandGroup>
                          {repos.map(repo => (
                            <CommandItem
                              value={repo.label}
                              key={repo.value}
                              onSelect={() => {
                                form.setValue('gitUrl', repo.value)
                                setOpen(false)
                              }}
                            >
                              <IconCheck
                                className={cn(
                                  'mr-2',
                                  repo.value === field.value
                                    ? 'opacity-100'
                                    : 'opacity-0'
                                )}
                              />
                              {repo.label}
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
          <FormField
            control={form.control}
            name="name"
            render={({ field }) => (
              <FormItem>
                <FormLabel required>Custom name</FormLabel>
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
              Create
            </Button>
          </div>
        </form>
        <FormMessage className="text-center" />
      </div>
    </Form>
  )
}

function resolveRepoNameFromUrl(path: string | undefined) {
  if (!path) return ''
  return path.split('/')?.pop()
}
