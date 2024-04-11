import * as React from 'react'
import { useCombobox } from 'downshift'
import type {
  UseComboboxReturnValue,
  UseComboboxState,
  UseComboboxStateChangeOptions
} from 'downshift'
import { isNil, omitBy } from 'lodash-es'
import Textarea from 'react-textarea-autosize'

import { cn } from '@/lib/utils'
import {
  Popover,
  PopoverAnchor,
  PopoverClose,
  PopoverContent,
  PopoverPortal
} from '@/components/ui/popover'

import { Input } from './ui/input'

interface SearchableSelectContextValue<T = any>
  extends UseComboboxReturnValue<T> {
  open: boolean
  anchorRef: React.RefObject<HTMLElement>
}

export const SearchableSelectContext = React.createContext(
  {} as SearchableSelectContextValue
)

export const SearchableSelectClose = PopoverClose
export const SearchableSelectAnchor = React.forwardRef<
  React.ElementRef<typeof PopoverAnchor>,
  React.ComponentPropsWithoutRef<typeof PopoverAnchor>
>((props, forwardRef) => {
  return <PopoverAnchor {...props} ref={forwardRef} />
})
SearchableSelectAnchor.displayName = 'SearchableSelectAnchor'

export const SearchableSelectTextarea = React.forwardRef<
  React.ElementRef<typeof Textarea>,
  React.ComponentPropsWithoutRef<typeof Textarea>
>((props, forwardRef) => {
  const { getInputProps } = React.useContext(SearchableSelectContext)
  const { onKeyDown, onChange, onInput, onBlur, onClick, ...rest } = props

  return (
    <Textarea
      {...getInputProps(
        omitBy(
          {
            onKeyDown,
            onChange,
            onInput,
            onBlur,
            onClick,
            ref: forwardRef
          },
          isNil
        )
      )}
      {...rest}
    />
  )
})
SearchableSelectTextarea.displayName = 'SearchableSelectTextarea'

export const SearchableSelectInput = React.forwardRef<
  React.ElementRef<typeof Input>,
  React.ComponentPropsWithoutRef<typeof Input>
>((props, forwardRef) => {
  const { getInputProps } = React.useContext(SearchableSelectContext)
  const { onKeyDown, onChange, onInput, onBlur, onClick, ...rest } = props

  return (
    <Input
      {...getInputProps(
        omitBy(
          {
            onKeyDown,
            onChange,
            onInput,
            onBlur,
            onClick,
            ref: forwardRef
          },
          isNil
        )
      )}
      // ref={inputRef as React.RefObject<HTMLInputElement>}
      {...rest}
    />
  )
})
SearchableSelectInput.displayName = 'SearchableSelectInput'

export const SearchableSelectContent = React.forwardRef<
  React.ElementRef<typeof PopoverContent>,
  React.ComponentPropsWithoutRef<typeof PopoverContent> & {
    popupMatchAnchorWidth?: boolean
  }
>(({ children, style, popupMatchAnchorWidth, ...rest }, forwardRef) => {
  const { getMenuProps, anchorRef } = React.useContext(SearchableSelectContext)
  const popupWidth = React.useRef<number | undefined>(undefined)

  React.useLayoutEffect(() => {
    if (popupMatchAnchorWidth) {
      const anchor = anchorRef.current
      if (anchor) {
        const rect = anchor.getBoundingClientRect()
        popupWidth.current = rect.width
      }
    }
  }, [])

  return (
    <PopoverPortal>
      <PopoverContent
        align="start"
        onOpenAutoFocus={e => e.preventDefault()}
        style={{
          width: popupWidth.current,
          ...style
        }}
        {...getMenuProps({ ref: forwardRef }, { suppressRefError: true })}
        {...rest}
      >
        {children}
      </PopoverContent>
    </PopoverPortal>
  )
})
SearchableSelectContent.displayName = 'SearchableSelectContent'

interface SearchableSelectOptionProps<T = any> {
  item: T
  index: number
  className?: string
  disabled?: boolean
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((p: { selected: boolean; highlighted: boolean }) => React.ReactNode)
}

export const SearchableSelectOption = React.forwardRef<
  React.RefObject<HTMLDivElement>,
  SearchableSelectOptionProps
>(({ item, index, className, children, disabled, ...rest }, forwardRef) => {
  const { highlightedIndex, selectedItem, getItemProps } = React.useContext(
    SearchableSelectContext
  )
  const highlighted = highlightedIndex === index
  const selected = selectedItem === item

  return (
    <SearchableSelectClose key={item.id} asChild>
      <div
        className={cn(
          'relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none',
          highlighted && 'bg-accent text-accent-foreground',
          selected && 'font-bold',
          disabled && 'pointer-events-none opacity-50',
          className
        )}
        {...getItemProps({
          item,
          index,
          onMouseLeave: e => e.preventDefault(),
          onMouseOut: e => e.preventDefault()
        })}
        {...rest}
      >
        {typeof children === 'function'
          ? children({ highlighted, selected })
          : children}
      </div>
    </SearchableSelectClose>
  )
})
SearchableSelectOption.displayName = 'SearchableSelectOption'

interface SearchableSelectProps<T> {
  options: T[] | undefined
  onSelect?: (data: T) => void
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((contextValue: SearchableSelectContextValue) => React.ReactNode)
  open?: boolean
  onOpenChange?: (v: boolean) => void
  stayOpenOnInputClick?: boolean
}

export function SearchableSelect<T extends { id: number | string }>({
  options,
  onSelect,
  children,
  open: propsOpen,
  onOpenChange,
  stayOpenOnInputClick
}: SearchableSelectProps<T>) {
  const anchorRef = React.useRef<HTMLElement>(null)

  const stateReducer = React.useCallback(
    (
      state: UseComboboxState<T>,
      actionAndChanges: UseComboboxStateChangeOptions<T>
    ) => {
      const { type, changes } = actionAndChanges
      switch (type) {
        case useCombobox.stateChangeTypes.MenuMouseLeave:
          return {
            ...changes,
            highlightedIndex: state.highlightedIndex
          }
        case useCombobox.stateChangeTypes.InputClick:
          const isOpen = stayOpenOnInputClick ? true : changes.isOpen
          return {
            ...changes,
            isOpen
          }
        default:
          return changes
      }
    },
    [stayOpenOnInputClick]
  )

  const comboboxValue = useCombobox({
    items: options ?? [],
    defaultIsOpen: propsOpen,
    onSelectedItemChange({ selectedItem }) {
      if (selectedItem) {
        onSelect?.(selectedItem)
        onOpenChange?.(false)
      }
    },
    onIsOpenChange: ({ isOpen }) => {
      const nextOpen = !!isOpen
      onOpenChange?.(nextOpen)
    },
    stateReducer
  })

  const { setHighlightedIndex, highlightedIndex } = comboboxValue
  const isOpen = isNil(propsOpen)
    ? comboboxValue.isOpen
    : comboboxValue.isOpen && propsOpen

  React.useEffect(() => {
    if (isOpen && !!options?.length && highlightedIndex === -1) {
      setHighlightedIndex(0)
    }
  }, [isOpen, options])

  const contextValue = React.useMemo(() => {
    return { ...comboboxValue, open: isOpen, anchorRef }
  }, [comboboxValue, isOpen, anchorRef])

  return (
    <SearchableSelectContext.Provider value={contextValue}>
      <Popover open={isOpen}>
        {typeof children === 'function' ? children(contextValue) : children}
      </Popover>
    </SearchableSelectContext.Provider>
  )
}
