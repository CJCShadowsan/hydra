import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { statusKeys } from '@/lib/query/query-keys'
import type { StatusPayload } from '@/lib/api/types'
import { env } from '@/lib/env'

const MIN_BACKOFF = 1000
const MAX_BACKOFF = 15_000
const MIN_STATUS_COMMIT_MS = 500

export function useStatusStream(options?: { enabled?: boolean }) {
  const queryClient = useQueryClient()
  const enabled = options?.enabled ?? true
  const esRef = useRef<EventSource | null>(null)
  const backoffRef = useRef<number>(MIN_BACKOFF)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const commitTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const lastCommitRef = useRef(0)
  const pendingStatusRef = useRef<StatusPayload | undefined>(undefined)

  useEffect(() => {
    if (!enabled) return
    let active = true

    function clearReconnectTimer() {
      if (timerRef.current === undefined) return
      clearTimeout(timerRef.current)
      timerRef.current = undefined
    }

    function clearCommitTimer() {
      if (commitTimerRef.current === undefined) return
      clearTimeout(commitTimerRef.current)
      commitTimerRef.current = undefined
    }

    function flushPendingStatus() {
      if (!active || pendingStatusRef.current === undefined) return
      queryClient.setQueryData(statusKeys.detail(), pendingStatusRef.current)
      pendingStatusRef.current = undefined
      lastCommitRef.current = Date.now()
    }

    function scheduleStatusCommit(payload: StatusPayload) {
      pendingStatusRef.current = payload
      const elapsed = Date.now() - lastCommitRef.current
      if (elapsed >= MIN_STATUS_COMMIT_MS) {
        clearCommitTimer()
        flushPendingStatus()
        return
      }

      if (commitTimerRef.current !== undefined) return
      commitTimerRef.current = setTimeout(() => {
        commitTimerRef.current = undefined
        flushPendingStatus()
      }, MIN_STATUS_COMMIT_MS - elapsed)
    }

    function closeEventSource() {
      const es = esRef.current
      if (!es) return
      es.onopen = null
      es.onmessage = null
      es.onerror = null
      es.close()
      esRef.current = null
    }

    function scheduleReconnect(connect: () => void) {
      if (!active || timerRef.current !== undefined) return

      timerRef.current = setTimeout(() => {
        timerRef.current = undefined
        if (!active) return
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
        connect()
      }, backoffRef.current)
    }

    function connect() {
      if (!active) return
      clearReconnectTimer()
      closeEventSource()

      const es = new EventSource(`${env.managementApiUrl}/api/events`)
      esRef.current = es

      es.onopen = () => {
        if (!active) return
        backoffRef.current = MIN_BACKOFF
      }

      es.onmessage = (event: MessageEvent) => {
        if (!active) return
        try {
          scheduleStatusCommit(JSON.parse(event.data as string) as StatusPayload)
        } catch (_) {
          void _
        }
      }

      es.onerror = () => {
        if (esRef.current === es) {
          closeEventSource()
        } else {
          es.close()
        }
        scheduleReconnect(connect)
      }
    }

    connect()

    return () => {
      active = false
      clearReconnectTimer()
      clearCommitTimer()
      pendingStatusRef.current = undefined
      closeEventSource()
    }
  }, [queryClient, enabled])
}
