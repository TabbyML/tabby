import { useEffect, useRef } from 'react'

interface PopupWindowManagementOptions {
  popupUrl: string
  width?: number
  height?: number
  onMessage?: (data: any) => any
}

class PopupWindowManagement {
  private popupUrl: string | undefined
  private width: number
  private height: number
  private onMessage: ((data: any) => any) | void
  private popup: Window | null

  constructor({
    popupUrl,
    width = 600,
    height = 700,
    onMessage
  }: PopupWindowManagementOptions) {
    this.popupUrl = popupUrl
    this.popup = null
    this.onMessage = onMessage
    this.width = width
    this.height = height
    window.addEventListener('message', this.handleMessage, false)
  }

  destory() {
    window.removeEventListener('message', this.handleMessage, false)

    this.close()
    this.onMessage = undefined
  }

  open() {
    if (this.popup && !this.popup.closed) {
      this.popup.focus()
      return
    }

    const left = window.screenX + (window.outerWidth - this.width) / 2
    const top = window.screenY + (window.outerHeight - this.height) / 2.5
    const windowFeatures = `toolbar=no, menubar=no, width=${this.width}, height=${this.height}, top=${top}, left=${left}`
    this.popup = window.open(this.popupUrl, 'PopupWindow', windowFeatures)
  }

  close() {
    if (this.popup && !this.popup.closed) {
      this.popup.close()
      this.popup = null
    }
  }

  getPopupUrl() {
    return this.popupUrl
  }

  private handleMessage = (event: MessageEvent) => {
    if (event.origin === window.origin) {
      if (event.source === this.popup) {
        const data = event.data
        this.onMessage?.(data)
      }
    }
  }
}

const usePopupWindow = (
  options: Omit<PopupWindowManagementOptions, 'popupUrl'>
) => {
  const managementRef = useRef<PopupWindowManagement | null>(null)

  const open = (url: string) => {
    debugger
    if (managementRef.current) {
      const popupUrl = managementRef.current.getPopupUrl()
      if (popupUrl === url) {
        managementRef.current.open()
      } else {
        managementRef.current.destory()
        managementRef.current = new PopupWindowManagement({
          ...options,
          popupUrl: url
        })
        managementRef.current.open()
      }
    } else {
      managementRef.current = new PopupWindowManagement({
        ...options,
        popupUrl: url
      })
      managementRef.current.open()
    }
  }

  const close = () => {
    managementRef.current?.close()
  }

  const destory = () => {
    if (managementRef.current) {
      managementRef.current.destory()
      managementRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      if (managementRef.current) {
        managementRef.current.destory()
        managementRef.current = null
      }
    }
  }, [])

  return {
    open,
    close,
    destory
  }
}

export { usePopupWindow, PopupWindowManagement }
