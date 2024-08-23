'use client'

import { Button } from '@/components/ui/button'
import { IconListFilter } from '@/components/ui/icons'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

import PresetDocTable from './preset-doc-table'

export default function PresetDoc() {
  return (
    <>
      <div className="my-4 flex justify-between">
        <div className="flex items-center gap-4">
          <div className="relative">
            <IconListFilter className="absolute left-1.5 top-2.5 text-muted-foreground" />
            <Input className="w-60 pl-6" />
          </div>
          <Select defaultValue="">
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent align="end">
              <SelectGroup>
                <SelectItem value="">All Status</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
        <Button>Create</Button>
      </div>
      <PresetDocTable />
    </>
  )
}
