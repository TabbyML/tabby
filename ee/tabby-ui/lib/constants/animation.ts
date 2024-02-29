import { AnimationProps } from 'framer-motion'

const DEFAULT_VARIANTS = {
  hidden: { opacity: 0, y: 10 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      type: 'spring',
      duration: 0.3
    }
  }
}

const DEFAULT_ANIMTATION: AnimationProps = {
  initial: 'hidden',
  animate: 'visible',
  variants: DEFAULT_VARIANTS
}

export { DEFAULT_ANIMTATION, DEFAULT_VARIANTS }
