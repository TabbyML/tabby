import React, { useEffect, useState } from 'react'

import { getPaginationItem } from '@/lib/utils/pagination'
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious
} from '@/components/ui/pagination'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'

interface QuickNavPaginationProps {
  className?: string
  page: number
  totalCount: number
  // default: 10
  pageSize?: number
  // default: false
  showQuickJumper?: boolean
  // default: false
  showSizeChanger?: boolean
  onChange?: (page: number, pageSize: number) => void
}

const PAGE_SIZE_OPTIONS = ['5', '10', '20', '30', '50']

export const QuickNavPagination: React.FC<QuickNavPaginationProps> = ({
  className,
  page: propsPage,
  totalCount,
  pageSize: propsPageSize = 10,
  showQuickJumper = false,
  showSizeChanger = false,
  onChange
}) => {
  const [page, setPage] = useState(propsPage)
  const [pageSize, setPageSize] = useState(propsPageSize)

  const totalPageCount = Math.ceil(totalCount / pageSize)

  const paginationPages = getPaginationItem(totalCount, page, pageSize)

  // sync page
  useEffect(() => {
    if (!!propsPage && propsPage !== page) {
      setPage(propsPage)
    }
  }, [propsPage])

  // sync pageSize
  useEffect(() => {
    if (!!propsPageSize && propsPageSize !== pageSize) {
      setPageSize(propsPageSize)
    }
  }, [propsPageSize])

  if (paginationPages.length <= 1) return null

  return (
    <Pagination className={className}>
      <PaginationContent>
        {showSizeChanger && (
          <div className="mr-2 flex items-center space-x-2">
            <span className="text-sm font-medium">Rows per page</span>
            <Select
              value={String(pageSize)}
              onValueChange={v => {
                onChange?.(page, +v)
              }}
            >
              <SelectTrigger className="h-8 w-[70px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                {PAGE_SIZE_OPTIONS.map(option => {
                  return (
                    <SelectItem value={option} key={option}>
                      {option}
                    </SelectItem>
                  )
                })}
              </SelectContent>
            </Select>
          </div>
        )}
        <PaginationItem>
          <PaginationPrevious
            disabled={page === 1}
            onClick={() => {
              if (page === 1) return

              const _page = page - 1
              setPage(_page)
              onChange?.(_page, pageSize)
            }}
          />
        </PaginationItem>
        {showQuickJumper && (
          <>
            {paginationPages.map((item, index) => {
              return (
                <PaginationItem
                  key={`${item}-${index}`}
                  onClick={() => {
                    if (typeof item === 'number') {
                      setPage(item)
                      onChange?.(item, pageSize)
                    }
                  }}
                >
                  {typeof item === 'number' ? (
                    <PaginationLink
                      className="cursor-pointer"
                      isActive={item === page}
                    >
                      {item}
                    </PaginationLink>
                  ) : (
                    <PaginationEllipsis />
                  )}
                </PaginationItem>
              )
            })}
          </>
        )}
        <PaginationItem>
          <PaginationNext
            disabled={page === totalPageCount}
            onClick={() => {
              if (page === totalPageCount) return

              const _page = page + 1
              setPage(_page)
              onChange?.(_page, pageSize)
            }}
          />
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  )
}
