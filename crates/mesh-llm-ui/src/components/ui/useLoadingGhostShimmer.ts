import { useEffect, useRef, type RefObject } from 'react'

const LOADING_GHOST_SHIMMER_SELECTOR = '[data-loading-ghost-shimmer]'

export function useLoadingGhostShimmer<TElement extends HTMLElement>(rootRef: RefObject<TElement | null>) {
  const animationsRef = useRef<Animation[]>([])

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const root = rootRef.current
    if (prefersReducedMotion || !root || typeof Element.prototype.animate !== 'function') return undefined

    const shimmerElements = Array.from(root.querySelectorAll<HTMLElement>(LOADING_GHOST_SHIMMER_SELECTOR))
    animationsRef.current = shimmerElements.map((element, index) =>
      element.animate(
        [
          { opacity: 0, transform: 'translateX(-130%)' },
          { opacity: 0.42, offset: 0.5, transform: 'translateX(65%)' },
          { opacity: 0, transform: 'translateX(260%)' }
        ],
        {
          delay: index * 70,
          duration: 2400,
          easing: 'ease-in-out',
          iterations: Number.POSITIVE_INFINITY
        }
      )
    )

    return () => {
      animationsRef.current.forEach((animation) => animation.cancel())
      animationsRef.current = []
    }
  }, [rootRef])
}
