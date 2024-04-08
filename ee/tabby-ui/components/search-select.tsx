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

interface SearchSelectContextValue<T = any> extends UseComboboxReturnValue<T> {
  open: boolean
  anchorRef: React.RefObject<HTMLElement>
}

export const SearchSelectContext = React.createContext(
  {} as SearchSelectContextValue
)

export const SearchSelectClose = PopoverClose
export const SearchSelectAnchor = React.forwardRef<
  React.ElementRef<typeof PopoverAnchor>,
  React.ComponentPropsWithoutRef<typeof PopoverAnchor>
>((props, forwardRef) => {
  return <PopoverAnchor {...props} ref={forwardRef} />
})
SearchSelectAnchor.displayName = 'SearchSelectAnchor'

export const SearchSelectTextarea = React.forwardRef<
  React.ElementRef<typeof Textarea>,
  React.ComponentPropsWithoutRef<typeof Textarea>
>((props, forwardRef) => {
  const { getInputProps } = React.useContext(SearchSelectContext)
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
SearchSelectTextarea.displayName = 'SearchSelectTextarea'

export const SearchSelectInput = React.forwardRef<
  React.ElementRef<typeof Input>,
  React.ComponentPropsWithoutRef<typeof Input>
>((props, forwardRef) => {
  const { getInputProps } = React.useContext(SearchSelectContext)
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
SearchSelectInput.displayName = 'SearchSelectInput'

export const SearchSelectContent = React.forwardRef<
  React.ElementRef<typeof PopoverContent>,
  React.ComponentPropsWithoutRef<typeof PopoverContent> & {
    popupMatchAnchorWidth?: boolean
  }
>(({ children, style, popupMatchAnchorWidth, ...rest }, forwardRef) => {
  const { getMenuProps, anchorRef } = React.useContext(SearchSelectContext)
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
SearchSelectContent.displayName = 'SearchSelectContent'

interface SearchSelectOptionProps<T = any> {
  item: T
  index: number
  className?: string
  disabled?: boolean
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((p: { selected: boolean; highlighted: boolean }) => React.ReactNode)
}

export const SearchSelectOption = React.forwardRef<
  React.RefObject<HTMLDivElement>,
  SearchSelectOptionProps
>(({ item, index, className, children, disabled, ...rest }, forwardRef) => {
  const { highlightedIndex, selectedItem, getItemProps } =
    React.useContext(SearchSelectContext)
  const highlighted = highlightedIndex === index
  const selected = selectedItem === item

  return (
    <SearchSelectClose key={item.id} asChild>
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
    </SearchSelectClose>
  )
})
SearchSelectOption.displayName = 'SearchSelectOption'

interface SearchSelectProps<T> {
  options: T[] | undefined
  onSelect?: (data: T) => void
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((contextValue: SearchSelectContextValue) => React.ReactNode)
  open?: boolean
  onOpenChange?: (v: boolean) => void
  stayOpenOnInputClick?: boolean
}

export function SearchSelect<T extends { id: number | string }>({
  options,
  onSelect,
  children,
  open: propsOpen,
  onOpenChange,
  stayOpenOnInputClick
}: SearchSelectProps<T>) {
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
    <SearchSelectContext.Provider value={contextValue}>
      <Popover open={isOpen}>
        {typeof children === 'function' ? children(contextValue) : children}
      </Popover>
    </SearchSelectContext.Provider>
  )
}
