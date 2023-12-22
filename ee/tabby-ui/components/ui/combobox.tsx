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

interface ComboboxContextValue<T = any> extends UseComboboxReturnValue<T> {
  open: boolean
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement>
  anchorRef: React.RefObject<HTMLElement>
}

export const ComboboxContext = React.createContext({} as ComboboxContextValue)

export const ComboboxClose = PopoverClose
export const ComboboxAnchor = React.forwardRef<
  React.ElementRef<typeof PopoverAnchor>,
  React.ComponentPropsWithoutRef<typeof PopoverAnchor>
>((props, forwardRef) => {
  return <PopoverAnchor {...props} ref={forwardRef} />
})
ComboboxAnchor.displayName = 'ComboboxAnchor'

export const ComboboxTextarea = React.forwardRef<
  React.ElementRef<typeof Textarea>,
  React.ComponentPropsWithoutRef<typeof Textarea>
>((props, forwardRef) => {
  const { getInputProps, open } = React.useContext(ComboboxContext)
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
ComboboxTextarea.displayName = 'ComboboxTextarea'

export const ComboboxContent = React.forwardRef<
  React.ElementRef<typeof PopoverContent>,
  React.ComponentPropsWithoutRef<typeof PopoverContent> & {
    popupMatchAnchorWidth?: boolean
  }
>(({ children, style, popupMatchAnchorWidth, ...rest }, forwardRef) => {
  const { getMenuProps, anchorRef } = React.useContext(ComboboxContext)
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
ComboboxContent.displayName = 'ComboboxContent'

interface ComboboxOptionProps<T = any> {
  item: T
  index: number
  className?: string
  disabled?: boolean
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((p: { selected: boolean; highlighted: boolean }) => React.ReactNode)
}

export const ComboboxOption = React.forwardRef<
  React.RefObject<HTMLDivElement>,
  ComboboxOptionProps
>(({ item, index, className, children, disabled, ...rest }, forwardRef) => {
  const { highlightedIndex, selectedItem, getItemProps } =
    React.useContext(ComboboxContext)
  const highlighted = highlightedIndex === index
  const selected = selectedItem === item

  return (
    <ComboboxClose key={item.id} asChild>
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
    </ComboboxClose>
  )
})
ComboboxOption.displayName = 'ComboboxOption'

interface ComboboxProps<T> {
  options: T[] | undefined
  onSelect?: (data: T) => void
  inputRef?: React.RefObject<HTMLTextAreaElement | HTMLInputElement>
  children?:
    | React.ReactNode
    | React.ReactNode[]
    | ((contextValue: ComboboxContextValue) => React.ReactNode)
}

export function Combobox<T extends { id: number }>({
  options,
  onSelect,
  inputRef: propsInputRef,
  children
}: ComboboxProps<T>) {
  const [manualOpen, setManualOpen] = React.useState(false)
  const internalInputRef = React.useRef<HTMLTextAreaElement | HTMLInputElement>(
    null
  )
  const inputRef = propsInputRef ?? internalInputRef
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
        default:
          return changes
      }
    },
    []
  )

  const comboboxValue = useCombobox({
    items: options ?? [],
    isOpen: manualOpen,
    onSelectedItemChange({ selectedItem }) {
      if (selectedItem) {
        onSelect?.(selectedItem)
        setManualOpen(false)
      }
    },
    onIsOpenChange: ({ isOpen }) => {
      if (!isOpen) {
        setManualOpen(false)
      }
    },
    stateReducer
  })

  const { setHighlightedIndex, highlightedIndex } = comboboxValue
  const open = manualOpen && !!options?.length

  React.useEffect(() => {
    if (open && !!options.length && highlightedIndex === -1) {
      setHighlightedIndex(0)
    }
    if (open && !options.length) {
      setManualOpen(false)
    }
  }, [open, options])

  React.useEffect(() => {
    if (options?.length) {
      setManualOpen(true)
    } else {
      setManualOpen(false)
    }
  }, [options])

  const contextValue = React.useMemo(() => {
    return { ...comboboxValue, open, inputRef, anchorRef }
  }, [comboboxValue, open, inputRef, anchorRef])

  return (
    <ComboboxContext.Provider value={contextValue}>
      <Popover open={open}>
        {typeof children === 'function' ? children(contextValue) : children}
      </Popover>
    </ComboboxContext.Provider>
  )
}
