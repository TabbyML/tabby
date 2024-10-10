import { Transition, Variant, Variants } from 'framer-motion'

export const initialVariants: Variant = { y: 30, opacity: 0 }

export const cardTransition: Transition = {
  ease: 'easeOut',
  duration: 0.5
}

export function getCardVariants(delay?: number): Variants {
  return {
    initial: {
      y: 32,
      opacity: 0
    },
    offscreen: {
      opacity: 0,
      y: -10,
      transition: cardTransition
    },
    onscreen: {
      opacity: 1,
      y: 0,
      transition: {
        ...cardTransition,
        delay
      }
    }
  }
}
